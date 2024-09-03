import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import StandardScaler

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device

def tokenize_data(df):
    labels = list(df.columns.values.tolist())
    df = df.melt(var_name='category', value_name='text')
    id2label={id:label for id,label in enumerate(labels)}
    label2id={label:id for id,label in enumerate(labels)}
    df["labels"]=df.category.map(lambda x: label2id[x.strip()])
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels), id2label=id2label, label2id=label2id)
    model.to(device)
    return df, tokenizer, model

if __name__ == "__main__":

    df = pd.read_csv("../resources/S1.tsv",sep="\t", index_col=False)
    tokenizer, model = tokenize_data(df)
    print (tokenizer)

