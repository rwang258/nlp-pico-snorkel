#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import os
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import pickle


# In[ ]:


# stores all files as series
files_as_series = {}
# stores first sentence of all files
first_sentence_file = {}
# stores all POS files as series
pos_files_as_series = {}


# get index of each token in the .txt file this token is from and returns all indexes as a list.
def get_indexes_as_series(tokens):
    return tokens.index.tolist()
    
# returns a list of this value of size n, where each value is the 
# length of the .txt file this token is from, where n is the length of the input tokens series
def get_len_as_series(tokens):
    temp = [len(tokens) for i in range(0, len(tokens))]
    return temp


def file_to_series(file_name):    
    # Source: https://www.geeksforgeeks.org/read-a-file-line-by-line-in-python/
    with open(file_name) as f:
        lines = [line.strip() for line in f]
    return pd.Series(lines)

# Strip PubMed IDs from file names.
strip_pmid = lambda x: x.split(".")[0]

def iter_token_dir(dir_name, df, label_dict, col_name = "Token", ext_name = ".tokens"):
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(ext_name): 
            
            series_file = file_to_series(directory.decode("utf-8") + file_name)
            
            pos_series_file = file_to_series(directory.decode("utf-8") + file_name.split(".")[0] + ".pos")
            
            files_as_series[file_name] = series_file
            pos_files_as_series[file_name.split(".")[0] + ".pos"] = pos_series_file
            
            token_index = get_indexes_as_series(series_file)
            file_len = get_len_as_series(series_file)
                        
            PMID = strip_pmid(file_name)
            df_file = pd.DataFrame({col_name: series_file,
                                    "File": [file_name] * len(series_file),
                                    "Gold": label_dict.get(PMID),
                                    "PMID": [PMID] * len(series_file),
                                    "token_index": token_index,
                                    "file_len": file_len
                                   })
            df = pd.concat([df, df_file])
        else:
            continue
    return df

def iter_label_dir(dir_name, ext_name = ".AGGREGATED.ann"):
    label_dict = dict()
    directory = os.fsencode(dir_name)
    for file in os.listdir(directory):
        file_name = os.fsdecode(file)
        if file_name.endswith(ext_name): 
            series_file = file_to_series(directory.decode("utf-8") + file_name)
            PMID = strip_pmid(file_name)
            label_dict[PMID] = series_file
        else:
            continue
    return label_dict


# get sentence index, sentence, and parts of speech of sentence, that token is in
def get_sentence_info(token_index, file_name):
        
    token_series = files_as_series[file_name]
    pos_series = pos_files_as_series[file_name.split(".")[0] + ".pos"]
    sentence = []
    pos_sentence = []
    sentence_index = 0
    
    i = token_index
    
    if token_series[i]=='.':
        sentence.insert(0, token_series[i])
        pos_sentence.insert(0, pos_series[i])
        i-=1
        
    while i>=0 and token_series[i]!='.':
        sentence.insert(0, token_series[i])
        pos_sentence.insert(0, pos_series[i])
        i-=1

    # index within sentence
    sentence_index = token_index - (i+1)
    i = token_index+1 if token_series[token_index]!='.' else token_index

    while i<len(token_series) and token_series[i]!='.':
        sentence.append(token_series[i])
        pos_sentence.append(pos_series[i])
        i+=1

    if token_index==0:
        first_sentence_file[file_name] = [x.lower() for x in sentence]
            
    return (token_series.tolist(), pos_series.tolist())


def get_abstract(x):
    s, ps = x
    return s
def get_pos_abstract(x):
    s, ps = x
    return ps


# tokens that are punctuation.
def is_punctuation(x):
    return False if x.Token.lower() in string.punctuation else True

# Iterate through directory to obtain all gold labels, 
# mapped to their respective file names.
label_dict = iter_label_dir("annotations/aggregated/starting_spans/interventions/train/")

# Iterate through directory to obtain all tokens,
# mapped to their respective file names.
# original tokens
df_orig = pd.DataFrame()
df_orig = iter_token_dir("documents/", df_orig, label_dict)

# get sentence related columns for each token
df_orig["sentence_info"] = df_orig.apply(lambda x : get_sentence_info(x["token_index"], x["File"]), axis=1)


df_orig["abstract"] = df_orig["sentence_info"].apply(get_abstract)
df_orig["pos_abstract"] = df_orig["sentence_info"].apply(get_pos_abstract)

df_orig = df_orig.drop("sentence_info", 1)


df_orig = df_orig.reset_index(drop=True)

df_orig = df_orig.head(708703)

df_orig.to_pickle('df_orig.pickle')


with open('first_sentence_file.pickle', 'wb') as handle:
    pickle.dump(first_sentence_file, handle, protocol=pickle.HIGHEST_PROTOCOL)

