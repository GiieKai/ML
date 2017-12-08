
import pandas as pd 
import numpy as np
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing import sequence
import pickle
#About model package
import gensim
import matplotlib.pyplot as plt
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model,Sequential
from keras import regularizers,initializers
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.convolutional import Conv1D,MaxPooling1D
from gensim.models import word2vec,Word2Vec
from keras.layers.wrappers import Bidirectional

#deal with label data
def dataload_label(label_path):
    data = []
    train_label = []
    with open(label_path,encoding="utf-8", errors="strict") as inputfile:
        for line in inputfile:
            train_label.append(float(line.strip().split('+++$+++')[0]))
            data.append(line.strip().split('+++$+++')[1])
    return data,train_label


# # In[3]:


#deal with no label data
def dataload_nolabel(no_label_path):
    data = []
    #train_label = []
    with open(no_label_path,encoding="utf-8", errors="strict") as inputfile:
        for line in inputfile:
            data.append(line.strip())
    return data


# # In[4]:


#deal with test data
def dataload_test(inputpath):
    test_text = [line.strip().split(',') for line in open(inputpath, encoding = 'utf-8-sig')]
    data = []
    for row in range(1,len(test_text)):
        data.append(str(test_text[row][1::]))
    return data


# # In[13]:


def textprcoessingforword2vec(input_data):
    sentences = []
    for i in range(len(input_data)):
        sentences.append(text_to_word_sequence(input_data[i],filters='\t\n'+"'",split=" "))
    return sentences


def modeltoword2vec(sentences,word_dim):
    model = Word2Vec(sentences,sg=1, size=word_dim, window=5, min_count=1)
    model.save('word2vec_no')

def wordtransformLstm(data_list,name= '',path = 'word2vec_no'):
    print("dealing with ----" + name + "---- wordtransformLstm")
    model = Word2Vec.load(path)
    data = data_list
    X =np.array([model[s] for s in data])
    #X_final = reshape_zero(X,word_dim=100,max_len_seq=40)

    return  X 

def reshape_zero(data,word_dim=100,max_len_seq=40):
    print("doing padding zero")
    return np.array([np.concatenate([np.array([0.0]*word_dim*(max_len_seq-row.shape[0])).reshape(max_len_seq-row.shape[0],word_dim),row]) for row in data])

def save_transform_npy(data,name = ''):
    print("save "+name + "in npy")
    np.save(name+'.npy',data)
    
def findmaxlen(data):
    Highest=max([len(x) for x in data])
    return Highest
    
# #deal with text and padding 
# # def textprcoessing(train_data,test_data,most_common_words):
# #     #Arguments:num_words: None or int. Maximum number of words to work with 
# #     #(if set, tokenization will be restricted to the top num_words most common words in the dataset).

# #     tokenizer = Tokenizer(num_words=most_common_words,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n'+"'", split=" ")
    
# #     tokenizer.fit_on_texts(train_data+test_data)
# #     vocab_size = len(tokenizer.word_index) + 1

# #     x_train_seq = tokenizer.texts_to_sequences(train_data)
# #     x_test_seq = tokenizer.texts_to_sequences(test_data)
    
# #     maxlen_x_train = len(max(x_train_seq,key=len))
    
# #     final_x_train_seq = sequence.pad_sequences(x_train_seq, maxlen=maxlen_x_train, dtype='int32',padding='pre', truncating='pre', value=0.)
# #     final_x_test_seq =sequence.pad_sequences(x_test_seq, maxlen=maxlen_x_train, dtype='int32',padding='pre', truncating='pre', value=0.)

# #     return final_x_train_seq,final_x_test_seq,maxlen_x_train,vocab_size

# # In[6]:


# #model arch.
# # def build_model(output_dim,most_common_words,max_seq_len,opt):

# #     model = Sequential()
# #     model.add(Embedding(output_dim=output_dim,
# #                         input_dim=most_common_words, 
# #                         input_length=None,embeddings_initializer = 'RandomNormal'))
# #     model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.3))
# #     model.add(Dense(256, activation='relu'))
# #     model.add(Dropout(0.4))

# #     model.add(Dense(128, activation='relu'))
# #     model.add(Dropout(0.5))
# #     model.add(Dense(1, activation='sigmoid'))
# #     model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
# #     model.summary()
# #     return model

def build_model(word2vec_dim,max_len_seq):
    model = Sequential()
    #input_shape=(max_len_seq,word_dim)
    model.add(Bidirectional(LSTM(256, input_shape=(max_len_seq,word2vec_dim),dropout=0.5, recurrent_dropout=0.3)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy',optimizer='Adam', metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    word2vec_dim = 128
    max_len_seq = 43
#-------trainng word2vec------
    test_path = sys.arg[1]
    
    x_load_test =  dataload_test(test_path)

    x_test = textprcoessingforword2vec(x_load_test)

    X_Test_temp = wordtransformLstm(x_test,"test")
    X_Test = reshape_zero(X_Test_temp,word2vec_dim,max_len_seq)
    #save_transform_npy(X_Test,"test")



    model = load_model('model_weight.h5')

    print("--------------predict--------------")
    #X_Test = np.load("test.npy")
    y_test = model.predict_classes(X_Test)


    Out_path = sys.arg[2]
    with open(Out_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_test):
            f.write('%d,%d\n' %(i, v))