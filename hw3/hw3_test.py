import os,sys
import argparse
from keras.models import load_model,Model,Sequential

import keras.backend as K
#from python_utils import *
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import normalize
from keras.models import model_from_json
import pandas as pd
x_test_path = sys.argv[1]

model = model_from_json(open('model_weights.json').read())
model.load_weights('model_weights.h5')

# def load DATA
#data_train = pd.read_csv('train.csv')
x_test = pd.read_csv(x_test_path)
#

def toint(s):
    return np.reshape([int(tok) for tok in s.split()], (48, 48))

#data_train.feature = data_train.feature.apply(toint) / 255
x_test.feature = x_test.feature.apply(toint) / 255
#x_train = np.stack(data_train.feature.as_matrix()).reshape((-1, 48, 48, 1))
x_test = np.stack(x_test.feature.as_matrix()).reshape((-1, 48, 48, 1))

#y_train = np_utils.to_categorical(y_train,7)


y_test = model.predict(x_test)

val_classes = y_test.argmax(axis=-1)

output_path = sys.argv[2]

df = pd.DataFrame({'label':val_classes.T, 'id': np.arange(val_classes.shape[0])}).to_csv(output_path, index = False)