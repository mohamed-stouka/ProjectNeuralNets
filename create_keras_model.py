# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import clean_data as cd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


#2019 uci news dataset 
dataset = 'uci-news-aggregator.csv'
cat_count, dt = cd.clean_data(dataset)

def tokenize_text(dataframe):
    """
    Convert text sentences to word tokens/
    Subsequently converting them to numerical sequences.
    """
    #using tokenizer library to convert text sentences to word tokens and subsequently converting them to numerical sequences
    tokenizer = Tokenizer(num_words=10000, lower=True)
    tokenizer.fit_on_texts(dataframe['TITLE'].values)
    X = tokenizer.texts_to_sequences(dataframe['TITLE'].values)
    X = pad_sequences(X, 10)
    #converting the dependent categorical variable using one-hot encoding to numerical array
    Y = pd.get_dummies(dt['CATEGORY']).values
    np.set_printoptions(threshold=sys.maxsize)
    #Splitting the dataset into training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 22)

    return (X_train, X_test, Y_train, Y_test)
#call tokenize_text function to get the traning and test sets
X_train, X_test, Y_train, Y_test = tokenize_text(dt)

def create_model(learning_rate, num_dense_layers, num_dense_nodes, activation):
    """
    Create the Neural Network model with specified Hyperparaeters.
    """
    
    seed = 22
    np.random.seed(seed)
    model = Sequential()
    model.add(Embedding(10000, 64,input_length = X_train.shape[1] ))
    model.add(SpatialDropout1D(0.7 ))
    model.add(Conv1D(64,5,padding='valid',activation=activation,strides=1))
    model.add(MaxPooling1D(pool_size=2 ))
    model.add(LSTM(64, activation =activation))#, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.5 ))
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i+1)
        # add dense layer
        model.add(Dense(num_dense_nodes,activation=activation,
                        name=name))
    model.add(Dense(4,activation='softmax'))
    adam = Adam(lr = learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer= adam ,metrics = ['accuracy'])
    
    return model
        