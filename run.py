
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from scripts import eda, tokenization, train

import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


path_to_data = "resources/"

onlyfiles = [f for f in listdir(path_to_data) if isfile(join(path_to_data, f))]
datafiles = [f for f in onlyfiles if f.lower().endswith(".xlsx")]
print ("Performing EDA")
dfs_list = eda.eda(datafiles, path_to_data)


# all dfs together
df = pd.concat(dfs_list)
df.fillna("-", inplace=True)
df.to_csv("{}/union.tsv".format(path_to_data), sep="\t")
print ("Performing tokenization")
df_melted, tokenizer, model = tokenization.tokenize_data(df)
print ("Training")
train_dataloader, val_dataloader, test_dataloader = train.prepare_data_for_training(df_melted, tokenizer)
trainer = train.train(model,train_dataloader, val_dataloader, test_dataloader)
train.save_model(trainer, tokenizer)


# one by one
#dfs_list = dfs_list[:1]
#print (dfs_list)
#for n, df in enumerate (dfs_list):
#    print (n)
#    print ("Performing tokenization")
#    df_melted, tokenizer, model = tokenization.tokenize_data(df)
#    print ("Training")
#    train_dataloader, val_dataloader, test_dataloader = train.prepare_data_for_training(df_melted, tokenizer)
#    trainer = train.train(model,train_dataloader, val_dataloader, test_dataloader)
#    train.save_model(trainer, tokenizer)



