#!/usr/bin/env python

import sys

from pynvml import *
import torch
import copy
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import datamol as dm

from molfeat.utils.converters import SmilesConverter
from molfeat.trans.pretrained import PretrainedHFTransformer

from glob import glob

from molfeat.trans.fp import FPVecTransformer

from tanimoto_gp import TanimotoGP

from lightgbm import LGBMRegressor

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

class DTset(Dataset):
    def __init__(self, smiles, y, mf_featurizer):
        super().__init__()
        self.smiles = smiles
        self.mf_featurizer = mf_featurizer
        self.y = torch.tensor(y).float()
        # here we use the molfeat mf_featurizer to convert the smiles to
        # corresponding tokens based on the internal tokenizer
        # we just want the data from the batch encoding object
        self.transformed_mols = self.mf_featurizer._convert(smiles)

    @property
    def embedding_dim(self):
        return len(self.mf_featurizer)

    @property
    def max_length(self):
        return self.transformed_mols.shape[-1]
    
    def __len__(self):
        return self.y.shape[0]
    
    def collate_fn(self, **kwargs):
        # the default collate fn self.mf_featurizer.get_collate_fn(**kwargs)
        # returns None, which should just concatenate the inputs
        # You could also use `transformers.default_data_collator` instead
        return self.mf_featurizer.get_collate_fn(**kwargs)
    
    def __getitem__(self, index):
        datapoint = dict((name, val[index]) for name, val in self.transformed_mols.items())
        datapoint["y"] = self.y[index]
        return datapoint

    
class AwesomeNet(torch.nn.Module):
    def __init__(self, mf_featurizer, hidden_size=128, dropout=0.1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        # we get the underlying model from the molfeat featurizer
        # here we fetch the "base" huggingface transformer model 
        # and not the wrapper around for MLM
        # this is principally to get smaller model and training efficiency
        base_pretrained_model = getattr(mf_featurizer.featurizer.model, mf_featurizer.featurizer.model.base_model_prefix)
        self.embedding_layer = copy.deepcopy(base_pretrained_model)
        self.embedding_dim = mf_featurizer.featurizer.model.config.hidden_size
        # given that we are not concatenating layers, the following is equivalent
        # self.embedding_dim = len(mf_featurizer)
        # we get the the pooling layer from the molfeat featurizer
        self.pooling_layer = mf_featurizer._pooling_obj
        self.hidden_layer = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(len(mf_featurizer), self.hidden_size),
            torch.nn.ReLU()
        )
        self.output_layer = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, *, y=None, **kwargs):
        # get embeddings
        x = self.embedding_layer(**kwargs)
        # we take the last hidden state
        # you could also set `output_hidden_states` to true above 
        # and take x["hidden_states"][-1] instead
        emb = x["last_hidden_state"]
        # run poolings
        h = self.pooling_layer(
            emb,
            kwargs["input_ids"],
            mask=kwargs.get('attention_mask'),
        )
        # run through our custom and optional hidden layer
        h = self.hidden_layer(h)
        # run through output layers to get logits
        return self.output_layer(h)

def train_test_llm(featurizer, train_dt, test_dt):
    BATCH_SIZE = 64 
    train_loader = DataLoader(train_dt, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dt.collate_fn())
    test_loader = DataLoader(test_dt, batch_size=BATCH_SIZE, shuffle=False, collate_fn=test_dt.collate_fn())

    DEVICE = "cuda"
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    PNA_AGGREGATORS = ['mean', 'min', 'max', 'std']
    PNA_SCALERS = ['identity', 'amplification', 'attenuation']

    model = AwesomeNet(featurizer, hidden_size=64, dropout=0.1, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    model = model.to(DEVICE).float()
    model = model.train()


    loss_per_epoch_list = []
    with tqdm(range(NUM_EPOCHS)) as pbar:
        for epoch in pbar:
            losses = []
            #print_gpu_utilization()
            for data in train_loader:
                for k,v in data.items():
                    data[k] = data[k].to(DEVICE)
                optimizer.zero_grad()
                out = model(**data)
                loss = loss_fn(out.squeeze(), data["y"])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            loss_per_epoch_list.append(np.mean(losses))
            pbar.set_description(f"Epoch {epoch} - Loss {np.mean(losses):.3f}")


    model.eval()
    test_y_hat = []
    test_y_true = []
    with torch.no_grad():
        for data in test_loader:
            for k,v in data.items():
                data[k] = data[k].to(DEVICE)
            out = model(**data)
            test_y_hat.append(out.detach().cpu().squeeze())
            test_y_true.append(data["y"])
    test_y_hat = torch.cat(test_y_hat).cpu().squeeze().numpy()
    test_y_true = torch.cat(test_y_true).cpu().squeeze().numpy()
    rmse = mean_squared_error(test_y_true, test_y_hat, squared=False)
    r2 = r2_score(test_y_true, test_y_hat)
    print(f"RMSE = {rmse:.2f} R**2 = {r2:.2f}")
    return [r2,rmse]

featurizer_list = [PretrainedHFTransformer("Roberta-Zinc480M-102M", notation="smiles", dtype=torch.float, preload=True),
                   PretrainedHFTransformer(kind="ChemBERTa-77M-MLM", pooling="bert", preload=True),
                   PretrainedHFTransformer("GPT2-Zinc480M-87M", notation="smiles", dtype=torch.float, preload=True)]
#featurizer = PretrainedHFTransformer(kind="ChemGPT-1.2B", notation="selfies", device="cuda:0")]

#df = pd.read_csv("https://raw.githubusercontent.com/PatWalters/yamc/main/data/JAK2.smi",sep=" ",names=["SMILES","Name","pIC50"])

res_list = []
num_folds = 10
ecfp_transformer = FPVecTransformer(kind='ecfp', dtype=float)


for infile_name in sorted(glob("../data/A*.smi")):
    print(infile_name)
    df = pd.read_csv(infile_name,sep=" ",names=["SMILES","Name","pIC50"])

    df['fp'] = list(ecfp_transformer(df.SMILES))

    for i in range(0,num_folds):
        train, test = train_test_split(df)

        tanimoto_gp = TanimotoGP()
        tanimoto_gp.fit(np.stack(train.fp),train.pIC50.values)
        gp_pred = tanimoto_gp.predict(np.stack(test.fp))[0]
        gp_r2 = r2_score(test.pIC50,gp_pred)
        gp_rmse = mean_squared_error(test.pIC50,gp_pred)
        print(gp_r2,gp_rmse)
        res = [infile_name,"GP",i,gp_r2,gp_rmse]
        res_list.append(res)

        lgbm = LGBMRegressor()
        lgbm.fit(np.stack(train.fp),train.pIC50)
        lgbm_pred = lgbm.predict(np.stack(test.fp))
        lgbm_r2 = r2_score(test.pIC50,lgbm_pred)
        lgbm_rmse = mean_squared_error(test.pIC50,lgbm_pred)
        print(lgbm_r2, lgbm_rmse)
        res = [infile_name,"LGBM",i,lgbm_r2,lgbm_rmse]
        res_list.append(res)
        
        for featurizer in featurizer_list:    
            torch.cuda.empty_cache()
            print_gpu_utilization()

            train_dt = DTset(train.SMILES.values, train.pIC50.values, featurizer)
            test_dt = DTset(test.SMILES.values, test.pIC50.values, featurizer)
            res = [infile_name, featurizer.kind, i] + train_test_llm(featurizer, train_dt, test_dt)
            res_list.append(res)

res_df = pd.DataFrame(res_list, columns=["Filename","Featurizer","Cycle","r2","rmse"])
res_df.to_csv("llm_compare.csv",index=False)
        






