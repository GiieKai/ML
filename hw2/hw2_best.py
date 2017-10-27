#import package
import os, sys
from random import shuffle #cut set 
import numpy as np
import pandas as pd 
from math import log, floor
import argparse #use cmd run code
import xgboost as xgb
import csv
#IO File
X_Row_path = sys.argv[1]
Y_Row_path = sys.argv[2]
X_train_path = sys.argv[3]
Y_train_path = sys.argv[4]
X_test_path = sys.argv[5]
X_train = pd.read_csv(X_train_path)
X_train = np.array(X_train.values)

Y_train = pd.read_csv(Y_train_path)
Y_train =  np.array(Y_train.values)

X_test = pd.read_csv(X_test_path)
X_test = np.array(X_test.values)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

#define split valid
def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    #do shuffle first, and then do split
    X_all, Y_all = _shuffle(X_all, Y_all)

    X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

    return X_train, Y_train, X_valid, Y_valid

#params setting
params={
'booster':'gbtree',
'objective':'binary:logistic', 
'num_class':1,
'gamma':0.05,
'max_depth':12,
'subsample':0.4,
'colsample_bytree':0.7,
'silent':1 ,
'eta': 0.005,
'seed':710,
'nthread':4,
}

plst = list(params.items())
#Using 10000 rows for early stopping. 
all_data_size = len(X_train)
#offset = int(all_data_size*0.6)  
offset = 20000
num_rounds = 500 # 迭代你次数
xg_X_test = xgb.DMatrix(X_test)

#print(offset)

# 划分训练集与验证集 
xg_X_train = xgb.DMatrix(X_train[:offset,:], label=Y_train[:offset,0])
xg_X_valid = xgb.DMatrix(X_train[offset:,:], label=Y_train[offset:])

# return 训练和验证的错误率
watchlist = [(xg_X_train, 'train'),(xg_X_valid, 'val')]

model = xgb.train(plst, xg_X_train, num_rounds, watchlist,early_stopping_rounds=100)
preds = model.predict(xg_X_test,ntree_limit=model.best_iteration)
b = preds > 0.5
predict_Y_lsit =b.astype(int)
#print(predict_Y_lsit)
Out_path = sys.argv[6]
with open(Out_path, 'w') as f:
    f.write('id,label\n')
    for i, v in  enumerate(predict_Y_lsit):
        f.write('%d,%d\n' %(i+1, v))