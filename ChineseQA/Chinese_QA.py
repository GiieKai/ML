
# coding: utf-8

# In[1]:


import numpy as np
import math
import sys
import string
import pickle
import json
import jieba
import gensim

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from gensim.models import word2vec, Word2Vec

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D


# In[2]:


class RawExample(object):
    pass

def read_train_json(path):
    with open(path, encoding = 'utf-8') as f:
        data = json.load(f)
    all_data = []
    examples = []
    context_list = []
    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            context_list.append((context))
            for qa in qas:
                question = qa["question"]
                answers = qa["answers"]
                question_id = qa["id"]        
                all_data.append([context,question,answers])
                for ans in answers:
                    answer_start = int(ans["answer_start"])
                    answer_text = ans["text"]
                    e = RawExample()
                    e.title = title
                    e.context_id = len(context_list) - 1
                    e.question = question
                    e.question_id = question_id
                    e.answer_start = answer_start
                    e.answer_text = answer_text
                    examples.append(e)
    return examples, context_list, all_data

def read_test_json(path):
    with open(path, encoding = 'utf-8') as f:
        data = json.load(f)
    examples = []
    context_list = []

    for topic in data["data"]:
        title = topic["title"]
        for p in topic['paragraphs']:
            qas = p['qas']
            context = p['context']
            context_list.append((context))
            for qa in qas:
                question = qa["question"]
                question_id = qa["id"]
        for qa in qas:
            e = RawExample()
            e.title = title
            e.context_id = len(context_list) - 1
            e.question = question
            e.question_id = question_id
            examples.append(e)
    return examples, context_list

class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.nwords = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<unk>')
    def add_word(self, w):
        if w not in self.word2idx:
            self.word2idx[w] = self.nwords
            self.idx2word.append(w)
            self.nwords += 1
    def __call__(self, w):
        if w not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[w]
    def __len__(self):
        return self.nwords
    def prepare_vocab(train_data):
        word_vocab = Vocabulary()
        char_vocab = Vocabulary()
        for i, context in enumerate(train_data[1]):
            context = context[0]
            for j, word in enumerate(jieba.lcut(context)):
                word_vocab.add_word(word)
                for k, char in enumerate(word):
                    char_vocab.add_word(char)
        for i, e in enumerate(train_data[0]):
            for j, question in enumerate(jieba.lcut(e.question)):
                word_vocab.add_word(word)
                for k, char in enumerate(word):
                    char_vocab.add_word(char)
        
        vocab_path = 'vocab'
        with open(vocab_path, 'wb') as f:
            pickle.dump((word_vocab,char_vocab), f)
            
def padding(seqs,word_number,vocab):
    length = max(word_number)
    for seq in seqs:
        for i in range(length-len(seq)):
            seq.append(vocab('<pad>'))
            
def get_answer(examples):
    answer_context = []
    answer_postion = []
    for e in examples:
        answer_context.append(e.answer_text)
        answer_postion.append((e.answer_start,e.answer_start+len(e.answer_text)-1))
    return answer_context, answer_postion

def get_question(examples):
    question_context = []
    for e in examples:
        question_context.append(e.question)
    return question_context

def sliding(data,window_size):
    best_start = []
    for context,question in (zip(data["context"],data["question"])):
        best_score = float(-np.inf)
        temp_start = 0
        for start in range(len(context)-window_size):    
            score = 0
            for i in (question):
                for j in range(window_size):
                    score += cosine_similarity(context[start+j],i)
            if score > best_score:
                temp_start = start
                best_score = score
        best_start.append(temp_start)
    return best_start

def f1_score(pred,truth):
    common = Counter(pred) & Counter(truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# In[3]:


# Read Data
train_path = 'C:/Users/HongLab/Desktop/KTLin/Final/Chinese QA/train-v1.1.json'
test_path = 'C:/Users/HongLab/Desktop/KTLin/Final/Chinese QA/test-v1.1.json'

train_data = read_train_json(train_path)
test_data = read_test_json(test_path)

answer_context, answer_postion = get_answer(train_data[0])
question_context = get_question(train_data[0])


# In[4]:


temp = []
for i in range(len(train_data[2])):
    temp.append(train_data[2][i][0])
    
Story = temp
Q = question_context
# A = answer_context

all_word = []

for i in range(len(Story)):
    for j in range(len(Story[i])):
        all_word.append(Story[i][j])
for i in range(len(Q)):
    for j in range(len(Q[i])):
        all_word.append(Q[i][j])
# for i in range(len(A)):
#     for j in range(len(A[i])):
#         all_word.append(A[i][j])

model_gensim = Word2Vec(all_word, sg=1, size=100, window=5, min_count=1)
#model_gensim.save('model_gensim')

#preprocessing
Story_vector = []
for i in range(len(Story)):
    Story_vector.append([])
    for j in range(len(Story[i])):
        Story_vector[i].append(model_gensim.wv[Story[i][j]])

Q_vector = []
for i in range(len(Q)):
    Q_vector.append([])
    for j in range(len(Q[i])):
        Q_vector[i].append(model_gensim.wv[Q[i][j]])
        
# A_vector = []
# for i in range(len(A)):
#     A_vector.append([])
#     for j in range(len(A[i])):
#         A_vector[i].append(model_gensim.wv[A[i][j]])

Story_max_len = 0
for i in range(len(Story_vector)):
    if len(Story_vector[i]) > Story_max_len:
        Story_max_len = len(Story_vector[i])

Q_max_len = 0
for i in range(len(Q_vector)):
    if len(Q_vector[i]) > Q_max_len:
        Q_max_len = len(Q_vector[i])
        
# A_max_len = 0
# for i in range(len(A_vector)):
#     if len(A_vector[i]) > A_max_len:
#         A_max_len = len(A_vector[i])
        
zeros = np.zeros((100,),dtype = np.float32)     
   
for i in range(len(Story_vector)):
    makeup_len = Story_max_len - len(Story_vector[i])        
    for j in range(makeup_len):
        Story_vector[i].append(zeros)
        
for i in range(len(Q_vector)):
    makeup_len = Q_max_len - len(Q_vector[i])        
    for j in range(makeup_len):
        Q_vector[i].append(zeros)
        
# for i in range(len(A_vector)):
#     makeup_len = A_max_len - len(A_vector[i])        
#     for j in range(makeup_len):
#         A_vector[i].append(zeros)


Q_vector = np.array(Q_vector)
Story_vector = np.array(Story_vector)

word_vector_weights = model_gensim.wv

# In[9]:


#img->>>>Story
def story_model(Story_vector, num_words, embedding_dim, seq_length):
    print("Creating Story model...")
    dropout_rate = 0.2
    model = Sequential()
#     model.add(Embedding(num_words, embedding_dim, 
#         weights=[Story_vector], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    
    model.summary()
    return model

def Word2VecModel(Q_vector, num_words,embedding_dim, seq_length):
    print("Creating text model...")
    dropout_rate = 0.2
    
    model = Sequential()
#     model.add(Embedding(num_words, embedding_dim, 
#         weights=[Q_vector], input_length=seq_length, trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    
    model.summary()
    return model

#Combine Story and Question
def vqa_model(Q_vector, Story_vector, num_words, seq_length, num_classes):
    vgg_model = story_model(Story_vector, num_words, 100, 1158)
    lstm_model = Word2VecModel(Q_vector, num_words, 100, 239)
    print("Merging final model...")
    dropout_rate = 0.2
   
    fc_model = Sequential()
    fc_model.add(Merge([vgg_model, lstm_model], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    fc_model.summary()
    return fc_model

#print(Q_vector.shape)
num_classes = 2
num_words = Story_vector.shape[1]
seq_length = 14611
model = vqa_model(Q_vector, Story_vector, num_words, seq_length, num_classes)
BATCH_SIZE = 1024
NBR_EPOCHS = 100

model.fit([Story_vector, Q_vector], answer_postion, batch_size=BATCH_SIZE,
            epochs=NBR_EPOCHS, validation_split=0.1)

model.save_weights('weights.h5')

