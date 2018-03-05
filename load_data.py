import pandas as pd
import os
import numpy as np
import random
from io import open
from textblob import TextBlob
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')
###
def strip_quotations_newline(text):
    text=str(text).encode('ascii', 'ignore').decode('ascii')
    text = text.rstrip()
    '''if text[0] == '"':
        text = text[1:]
    if text[-1] == '"':
        text = text[:-1]'''
    return text

def expand_around_chars(text, characters):
    for char in characters:
        text = text.replace(char, " "+char+" ")
    return text

def split_text(text):
    text=str(text).encode('ascii', 'ignore').decode('ascii')
    text = strip_quotations_newline(text)
    text = expand_around_chars(text, '".,()[]{}:;')
    splitted_text = text.split(" ")
    cleaned_text = [x for x in splitted_text if len(x)>1]
    text_lowercase = [x.lower() for x in cleaned_text]
    return text_lowercase

###
def pow10(x):
    i = 1;
    while((i * 10) < x):
        i *= 10.0;
    return i
    
def normalize_col(col1, method):
    cc_mean = np.mean(col1)
    if method == 'pow10':         
        return col1 / pow10(np.max(col1))
    else:
        return col1 - cc_mean
    
def normalize_matrix(X, method = 'mean'):
    no_rows, no_cols = np.shape(X)
    X_normalized = np.zeros(shape=(no_rows, no_cols))
    X_normalized[:,0] = X[:,0]
    for ii in range(1,no_cols):
        X_normalized[:, ii] = normalize_col(X[:, ii], method)
    return X_normalized    

###
def pos_tag(txt):
    txt=str(txt).encode('ascii', 'ignore').decode('ascii')
    blob=TextBlob(txt)
    return blob.tags

def amazon_reviews():
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    data=pd.read_csv('Amazon_Unlocked_Mobile.csv',encoding='utf-8')
    for line in data['Reviews'][10000:20000]:
        Y_test.append('Review')
        X_test.append(split_text(line))
    for line in data['Reviews'][10000:20000]:
        Y_train.append('Review')
        X_train.append(split_text(line))
    '''datafolder = './datasets/amazon/'
    files = os.listdir(datafolder)
    Y_train, Y_test, X_train, X_test,  = [], [], [], []
    for file in files:
        f = open(datafolder + file, 'r', encoding="utf8")
        label = file
        lines = f.readlines()
        no_lines = len(lines)
        no_training_examples = int(0.7*no_lines)
        for line in lines[:no_training_examples]:
            Y_train.append(label)
            X_train.append(split_text(line))
        f.close()'''
    data['processed_Reviews']=data['Reviews'][:10000].apply(pos_tag)
    processed_data=data.dropna()#remove null values
    processed=processed_data.drop(['Reviews'],axis=1)
    processed.to_csv('preprocessed_reviews.csv')
    #print  X_train, Y_train, X_test, Y_test
    return X_train, Y_train, X_test, Y_test
