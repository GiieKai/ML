
# coding: utf-8

# In[1]:


from keras.models import Model,Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
def build_model():
    input_img = Input(shape=(48, 48, 1))

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = MaxPooling2D(pool_size=(5, 5), strides=(2, 2))(block1)
    block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(64, (3, 3), activation='relu')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)

    block3 = Conv2D(64, (3, 3), activation='relu')(block2)
    block3 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block3)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)
    block3 = Dropout(0.5)(block3)
    
    
    block4 = Conv2D(128, (3, 3), activation='relu')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    block4 = Dropout(0.5)(block4)
    
    block5 = Conv2D(128, (3, 3), activation='relu')(block4)
    block5 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block5)
    block5 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block5)
    block5 = Flatten()(block5)

    fc1 = Dense(1024, activation='relu')(block5)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(1024, activation='relu')(fc1)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    #opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model



# In[ ]:


import numpy as np
import pandas as pd
import argparse

import os,sys

data_path = sys.argv[1]
# def load DATA
data_train = pd.read_csv(data_path)
#x_test = pd.read_csv('test.csv')
y_train = data_train.iloc[:,0]
#print(y_train)
def toint(s):
    return np.reshape([int(tok) for tok in s.split()], (48, 48))
data_train.feature = data_train.feature.apply(toint) / 255
#x_test.feature = x_test.feature.apply(toint) / 255
x_train = np.stack(data_train.feature.as_matrix()).reshape((-1, 48, 48, 1))
#x_test = np.stack(x_test.feature.as_matrix()).reshape((-1, 48, 48, 1))

y_train = np_utils.to_categorical(y_train,7)


# In[ ]:


# # 切20%拿來畫圖
# vaild = int(len(x_train)*0.2)
# x_train_matrix = x_train[vaild:]
# y_train_matrix = y_train[vaild:]
# #前20個拿來當作test
# x_vaild_matrix =  x_train[:vaild]
# y_vaild_matrix =  y_train[:vaild]


# # In[ ]:


# x_train_matrix.shape


# In[ ]:


model = build_model()
from math import log
labels_dict = {0: 3995, 1: 436, 2: 4097, 3: 7215, 4: 4830, 5: 3171, 6: 4966}
labels_dict_log = [0]*7


# In[ ]:


for i in range(len(labels_dict)):
    labels_dict_log[i] = log(labels_dict[i])
    labels_dict[i] = labels_dict_log[i] 


# In[ ]:
train_result = model.fit(x_train,y_train, epochs = 100, batch_size = 256 ,shuffle='True',validation_split = 0.2, class_weight=labels_dict )

#train_result = model.fit(x_train_matrix,y_train_matrix, epochs = 100, batch_size = 256 ,shuffle='True',validation_split = 0.1, class_weight=labels_dict )


# In[ ]:


# import matplotlib.pyplot as plt
# # summarize history for accuracy
# plt.plot(train_result.history['acc'])
# plt.plot(train_result.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='lower right')
# plt.show()


# In[ ]:


# pridect_y =  model.predict(x_vaild_matrix)


# # In[ ]:


# y_train_matrix_predict_val_classes = pridect_y.argmax(axis=-1)


# # In[ ]:


# y_train_matrix_row_val_classes = y_vaild_matrix.argmax(axis=-1)


# # In[ ]:


# print("\t[Info] Display Confusion Matrix:")  
# print(pd.crosstab(y_train_matrix_predict_val_classes, y_train_matrix_row_val_classes, rownames=['label'], colnames=['predict'],margins = True)) 

# save model
model_json = model.to_json()
# save structure
open('model_weights.json', 'w').write(model_json)
# save weight
model.save_weights('model_weights.h5')