#0814

import pandas as pd 
import numpy as np
from random import shuffle

def _shuffle(X):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize])

def load_data_ratings(path):
    ratings = pd.read_csv(path, encoding='latin-1', usecols=['TrainDataID','UserID', 'MovieID', 'Rating'])
    del ratings['TrainDataID']
    nplist = np.array(ratings)
    nplist = _shuffle(nplist)
    X_user = nplist[:,0]
    X_movie = nplist[:,1]
    Y = nplist[:,2]
    
    return X_user,X_movie,Y,ratings



def load_data_user(path):
    users = pd.read_csv(path , sep='::', encoding='latin-1', usecols=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
    print (len(users), 'descriptions of', max_userid,'users loaded.')
    return users

def load_data_movies(path):
    movies = pd.read_csv(path, sep='::', encoding='latin-1', usecols=['movieID', 'Title', 'Genres'])
    print (len(movies), 'descriptions of', max_movieid, 'movies loaded.')
    return movies


def one_hot_movie_Genres(dfname,colsname):
    movietype = list(dfname[colsname])
    new = []
    tt = []
    for i in range(len(movietype)):
        new.append([])
        new[i] = movietype[i].split('|')
        for j in range(len(new[i])):
            tt.append(new[i][j])

    newnew = []
    for i in range(len(new)):
        newnew.append([])
        moviett = [0]*len(jj)
        for j in range(len(new[i])):
            ind = kk.index(new[i][j])
            moviett[ind] = 1
        newnew[i]=moviett

    return newnew


def one_hot_user_Occupation(dfname,colsname):
    user_one_hot = []
    #oneone = [0]*len(set(users['Occupation']))
    for i in range(len(users)):
        oneone = [0]*len(set(users['Occupation']))
        oneone[users['Occupation'][i]] = 1
        user_one_hot.append(oneone)

    return user_one_hot


def load_data_test(path):
    test = pd.read_csv(test_path, encoding='latin-1', usecols=["TestDataID","UserID","MovieID"])
    test_user = test['UserID']
    test_movie = test['MovieID']
    return [test_user,test_movie]  


import keras

from keras.layers import  Input

from keras.layers import Embedding

#keras.layers.core.Flatten()
from keras.layers.core import  Flatten

# keras.layers.Dot
from keras.layers import Dot

from keras.layers import Add
from keras.layers import Concatenate
from keras.layers import Average
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout
import keras.backend as K
#must do suffle first
from keras.regularizers import l2
from keras.layers.core import Dense

def rmse(y_true, y_pred):
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))


def get_model(n_users, n_items, latent_dim = 1024):
    user_input = Input(shape=[1])
    item_input = Input(shape = [1])
    #input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    #https://faroit.github.io/keras-docs/2.0.8/layers/embeddings/#embedding
    user_vec_dot = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_dot = Flatten()(user_vec_dot)
    
    item_vec_dot = Embedding(input_dim = n_items+1, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_dot = Flatten()(item_vec_dot)
    
    user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim)(user_input)
    #user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_Con = Flatten()(user_vec_Con)
    #user_vec_Con = Dropout(0.1)(user_vec_Con)


    item_vec_Con = Embedding(input_dim = n_items+1, output_dim= latent_dim)(item_input)
    #item_vec_Con = Embedding(input_dim = n_items, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_Con = Flatten()(item_vec_Con)
    #item_vec_Con = Dropout(0.1)(item_vec_Con)

    # user_bias = Embedding(input_dim = n_users, output_dim= 1, embeddings_initializer="zeros")(user_input)
    user_bias = Embedding(input_dim = n_users, output_dim= 1)(user_input)
    user_bias = Flatten()(user_bias)
    
    # item_bias = Embedding(input_dim = n_items+1, output_dim= 1, embeddings_initializer="zeros")(item_input)       
    item_bias = Embedding(input_dim = n_items+1, output_dim= 1)(item_input)

    item_bias = Flatten()(item_bias)
    
    
    r_hot = Dot(axes=1)([user_vec_dot,item_vec_dot])
    r_hot = Add()([r_hot,user_bias, item_bias])

    embedding_Con = Concatenate()([user_vec_Con, item_vec_Con,r_hot])
    hidden_1 = Dense(64,activation = 'linear',kernel_regularizer=l2(5))(embedding_Con)
    hidden_1 = Dropout(0.5)(hidden_1)

    # hidden_2 = Dense(32,activation = 'linear',kernel_regularizer=l2(10))(hidden_1)
    # hidden_2 = Dropout(0.25)(hidden_2)

    pred = Dense(1, activation='linear',kernel_regularizer=l2(25))(hidden_1)

    # outt = Concatenate()([r_hot,pred])
    # outt = Dense(1, activation='linear',kernel_regularizer=l2(1))(outt)


    model = keras.models.Model([user_input, item_input], pred)
    model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
    return model

def get_model_MF(n_users, n_items, latent_dim = 128):
    user_input = Input(shape=[1])
    item_input = Input(shape = [1])
    #input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    #https://faroit.github.io/keras-docs/2.0.8/layers/embeddings/#embedding
    user_vec_dot = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_dot = Flatten()(user_vec_dot)
    
    item_vec_dot = Embedding(input_dim = n_items+1, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_dot = Flatten()(item_vec_dot)
    
    # user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim)(user_input)
    # #user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    # user_vec_Con = Flatten()(user_vec_Con)
    # #user_vec_Con = Dropout(0.1)(user_vec_Con)


    # item_vec_Con = Embedding(input_dim = n_items+1, output_dim= latent_dim)(item_input)
    # #item_vec_Con = Embedding(input_dim = n_items, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    # item_vec_Con = Flatten()(item_vec_Con)
    # #item_vec_Con = Dropout(0.1)(item_vec_Con)

    user_bias = Embedding(input_dim = n_users, output_dim= 1, embeddings_initializer="zeros")(user_input)
    #user_bias = Embedding(input_dim = n_users, output_dim= 1)(user_input)
    user_bias = Flatten()(user_bias)
    
    item_bias = Embedding(input_dim = n_items+1, output_dim= 1, embeddings_initializer="zeros")(item_input)       
    #item_bias = Embedding(input_dim = n_items+1, output_dim= 1)(item_input)

    item_bias = Flatten()(item_bias)
    
    
    r_hot = Dot(axes=1)([user_vec_dot,item_vec_dot])
    r_hot = Add()([r_hot,user_bias, item_bias])

    # embedding_Con = Concatenate()([user_vec_Con, item_vec_Con,r_hot])
    # hidden_1 = Dense(64,activation = 'linear',kernel_regularizer=l2(5))(embedding_Con)
    # hidden_1 = Dropout(0.5)(hidden_1)

    # hidden_2 = Dense(32,activation = 'linear',kernel_regularizer=l2(10))(hidden_1)
    # hidden_2 = Dropout(0.25)(hidden_2)

    # pred = Dense(1, activation='linear',kernel_regularizer=l2(25))(hidden_1)

    # outt = Concatenate()([r_hot,pred])
    # outt = Dense(1, activation='linear',kernel_regularizer=l2(1))(outt)


    model = keras.models.Model([user_input, item_input], r_hot)
    # model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
    return model 


def get_model_MF_nobias(n_users, n_items, latent_dim = 128):
    user_input = Input(shape=[1])
    item_input = Input(shape = [1])
    #input_dim: int > 0. Size of the vocabulary, i.e. maximum integer index + 1.
    #https://faroit.github.io/keras-docs/2.0.8/layers/embeddings/#embedding
    user_vec_dot = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    user_vec_dot = Flatten()(user_vec_dot)
    
    item_vec_dot = Embedding(input_dim = n_items+1, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    item_vec_dot = Flatten()(item_vec_dot)
    
    # user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim)(user_input)
    # #user_vec_Con = Embedding(input_dim = n_users, output_dim= latent_dim, embeddings_initializer="random_normal")(user_input)
    # user_vec_Con = Flatten()(user_vec_Con)
    # #user_vec_Con = Dropout(0.1)(user_vec_Con)


    # item_vec_Con = Embedding(input_dim = n_items+1, output_dim= latent_dim)(item_input)
    # #item_vec_Con = Embedding(input_dim = n_items, output_dim= latent_dim, embeddings_initializer="random_normal")(item_input)
    # item_vec_Con = Flatten()(item_vec_Con)
    # #item_vec_Con = Dropout(0.1)(item_vec_Con)

    user_bias = Embedding(input_dim = n_users, output_dim= 1, embeddings_initializer="zeros")(user_input)
    #user_bias = Embedding(input_dim = n_users, output_dim= 1)(user_input)
    user_bias = Flatten()(user_bias)
    
    item_bias = Embedding(input_dim = n_items+1, output_dim= 1, embeddings_initializer="zeros")(item_input)       
    #item_bias = Embedding(input_dim = n_items+1, output_dim= 1)(item_input)
    item_bias = Flatten()(item_bias)
    
    
    r_hot = Dot(axes=1)([user_vec_dot,item_vec_dot])
    r_hot = Add()([r_hot,user_bias, item_bias])

    # embedding_Con = Concatenate()([user_vec_Con, item_vec_Con,r_hot])
    # hidden_1 = Dense(64,activation = 'linear',kernel_regularizer=l2(5))(embedding_Con)
    # hidden_1 = Dropout(0.5)(hidden_1)

    # hidden_2 = Dense(32,activation = 'linear',kernel_regularizer=l2(10))(hidden_1)
    # hidden_2 = Dropout(0.25)(hidden_2)

    # pred = Dense(1, activation='linear',kernel_regularizer=l2(25))(hidden_1)

    # outt = Concatenate()([r_hot,pred])
    # outt = Dense(1, activation='linear',kernel_regularizer=l2(1))(outt)


    model = keras.models.Model([user_input, item_input], r_hot)
    # model.compile(loss = 'mse', optimizer = 'adam', metrics=['mse'])
    return model 

# from sklearn.manifold import TSNE
# from matplotlib import pyplot as plt
import sys
if __name__ == '__main__':
    # loading data 
    ratings_path = 'train.csv'
    X_user,X_movie,Y,ratings = load_data_ratings(ratings_path)
    #find the max_user and movie
    max_userid = ratings['UserID'].drop_duplicates().max()
    max_movieid = ratings['MovieID'].drop_duplicates().max()
    
    model = get_model_MF(max_userid, max_movieid)
    model.summary()
    # Use = ratings["UserID"]
    # Item = ratings["MovieID"]
    # Y = ratings["Rating"]
    # X = [Use,Item]

    # # X, Y = _shuffle(X,Y)

    # checkpoint = ModelCheckpoint('result/model_rmse{val_rmse:.4f}_epoch{epoch:03d}.hdf5',monitor='val_rmse', verbose=0,save_best_only=True, save_weights_only=False, period=10)

    # model.compile(loss='mse', optimizer='adam', metrics=[rmse])
    # model.fit([X_user,X_movie],Y, batch_size=1024, epochs=250, verbose=1, callbacks=[checkpoint], validation_split=0.1, validation_data=None
    #       , shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)

    #test.....
    test_path = sys.argv[1]

    model.load_weights("MF.hdf5")

    test_X = load_data_test(test_path)
    prediction = model.predict(test_X)
    Out_path = sys.argv[2]

    user_path = sys.argv[3]
    movie_path = sys.argv[4]

    with open(Out_path, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i, v in  enumerate(prediction):
            if v <1:
                v = 1
            elif v>5:
                v = 5
            f.write('%d,%.20f\n' %(i+1, v))

    # def draw(x,y):
    #     y = np.array(y)
    #     x = np.array(x,dtype=np.float64)
    #     # perform t-SNM embedding
    #     # vis_data = bh_sne(x)
    #     vis_data = TSNE(n_components=2).fit_transform(x)
    #     #plot the result
    #     vis_x = vis_data[:, 0]
    #     vis_y = vis_data[:, 1]
        
    #     print(vis_x.shape)
    #     print(vis_y.shape)
        
    #     cm = plt.cm.get_cmap('RdYlBu')
    #     sc = plt.scatter(vis_x, vis_y, c=y, cmap=cm)
    #     plt.colorvar(sc)
    #     plt.show()


    # ratings_path_movies = './data/movies.csv'
    # movies = load_data_movies(ratings_path_movies)

    # MoviesType = []
    # for i in range(movies.Genres.shape[0]):
    #     line = movies.Genres[i].strip().split('|')
    #     for j in range(len(line)):
    #         MoviesType.append(line[j])
            
    # UniqueType = np.unique(MoviesType)

    # class_movie = []
    # for i in range(6):
    #     class_movie.append(UniqueType[i:i+3])
        
    # tag = list(i for i in range(6))    
    # embedding = []
    # classification = []


    # user_emb = np.array(model.layers[2].get_weights()).squeeze()
    # print('user embedding shape:',user_emb.shape)
    # movie_emb = np.array(model.layers[3].get_weights()).squeeze()
    # print('movie embedding shape:',movie_emb.shape)
    # np.save('user_emb.npy', user_emb)
    # np.save('movie_emb.npy', movie_emb)


    # for i in range(movies.Genres.shape[0]):
    #     try:
    #         genres = movies.Genres[i].strip().split('|')
    #         if len(set(genres)&set(class_movie[0])) != 0:
    #             classification.append(tag[0])
    #             embedding.append(movie_emb[i])
    #         elif len(set(genres)&set(class_movie[1])) != 0:
    #             classification.append(tag[1])
    #             embedding.append(movie_emb[i])
    #         elif len(set(genres)&set(class_movie[2])) != 0:
    #             classification.append(tag[2])
    #             embedding.append(movie_emb[i])
    #         elif len(set(genres)&set(class_movie[3])) != 0:
    #             classification.append(tag[3])
    #             embedding.append(movie_emb[i])
    #         elif len(set(genres)&set(class_movie[4])) != 0:
    #             classification.append(tag[4])
    #             embedding.append(movie_emb[i])
    #         elif len(set(genres)&set(class_movie[5])) != 0:
    #             classification.append(tag[5])
    #             embedding.append(movie_emb[i])
    #     except ValueError:
    #         pass
            
    # draw(embedding, classification)