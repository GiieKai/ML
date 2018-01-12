
# coding: utf-8

# In[1]:


import numpy as np
import csv
import random
import math
import sys
import pandas as pd
import string
import pickle
import keras
import keras.backend as K
from matplotlib import pyplot as plt
from numpy.linalg import inv
from PIL import Image

from sklearn import cluster
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from collections import Counter

import sys
# In[2]:


# Read Data
test_path = sys.argv[2]
data = pd.read_csv(test_path)
data_1 = data['image1_index']
data_2 = data['image2_index']
image_path = sys.argv[1]
image = np.load(image_path)

image = image.astype('float32') / 255.
image = image.reshape((len(image), -1))
# print (image.shape)



# In[4]:

#128
encoding_dim = 128

input_img = Input(shape=(784,))

encoded = Dense(512, activation='relu')(input_img)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)

encoded = Dense(encoding_dim)(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(784, activation='tanh')(decoded)



autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

autoencoder.summary()

encoder.summary()

callbacks = []
# callbacks.append(EarlyStopping(monitor='val_rmse', patience=5))
# callbacks.append(ModelCheckpoint('model.h5', monitor='val_rmse', save_best_only=True))

autoencoder.compile(optimizer='adam', loss='mse')

#autoencoder.fit(image, image, epochs=150, batch_size=512, shuffle=True, validation_split=0.1, callbacks=callbacks)

#autoencoder.save('autoencoder.h5')


# In[5]:
autoencoder.load_weights('dnn.h5')

encoded_imgs = encoder.predict(image)
decoded_imgs = autoencoder.predict(image)

clforimg = cluster.KMeans(init='k-means++', n_clusters=2)
clforimg.fit(encoded_imgs)
ans = clforimg.fit_predict(encoded_imgs)
result = []
for i in range(len(data)):
    if ans[data_1[i]]==ans[[data_2[i]]]:
        result.append(1)
    else:
        result.append(0)

Out_path = sys.argv[3]
with open(Out_path, 'w') as f:
    f.write('ID,Ans\n')
    for i, v in  enumerate(result):    
        f.write('%d,%d\n' %(i, v))
# In[6]:


# def f1_score(pred,truth):
#     common = Counter(pred) & Counter(truth)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred)
#     recall = 1.0 * num_same / len(truth)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1


# In[7]:


# img_idx = 705

# img_idx1 = testImage1[img_idx]
# img_idx2 = testImage2[img_idx]

# plot = image[img_idx1]
# plot = plot.reshape(28, 28)
# plt.matshow(plot, cmap = plt.get_cmap('gray'))
# plt.show()

# plot = image[img_idx2]
# plot = plot.reshape(28, 28)
# plt.matshow(plot, cmap = plt.get_cmap('gray'))
# plt.show()

# # plot1 = decoded_imgs[img_idx]
# # plot1 = plot1.reshape(28, 28)
# # plt.matshow(plot1, cmap = plt.get_cmap('gray'))
# # plt.show()

# print (clusters[img_idx1]==clusters[img_idx2])
# result[img_idx]


# # in order to plot in a 2D figure
# encoding_dim = 128
# # this is our input placeholder
# input_img = Input(shape=(784,))

# # encoder layers
# encoded = Dense(512, activation='relu')(input_img)
# encoded = Dense(256, activation='relu')(encoded)
# encoded = Dense(128, activation='relu')(encoded)
# encoder_output = Dense(encoding_dim)(encoded)

# # decoder layers
# decoded = Dense(128, activation='relu')(encoder_output)
# decoded = Dense(256, activation='relu')(decoded)
# decoded = Dense(512, activation='relu')(decoded)
# decoded = Dense(784, activation='tanh')(decoded)