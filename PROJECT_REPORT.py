"""This file contains the stub code of CMSE802 Project.

It contains classes of slim LSTM which will be used to evaluate the computed hyperparameters.
"""
import warnings
from keras.layers import LSTM
from keras import backend as K
from keras import regularizers
from keras.optimizers import Adam
from keras.layers import LSTMCell
from keras.models import Sequential
from keras.layers import Dense,Embedding,SpatialDropout1D,Conv1D,MaxPooling1D,Bidirectional,Dropout

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

################################################################
class LSTMCell1(LSTMCell):
    """Original LSTM with non reduced parameters."""
    def build(self) :
        """
        Construct the LSTM with the correct parameters.
        Input is the required states.
        """
        pass
class LSTMCell3(LSTMCell):
    """
    LSTM with the required reduced parameters.
    Each gate is computed using only the bias.
    """
    def build(self):
        """
        Construct the LSTM with the correct parameters.
        Input is the required states.
        """
        pass
class LSTMCell4(LSTMCell):
    """
    LSTM with the required reduced parameters.
    Each gate is computed using only the previous hidden state but with point wise multiplication.
    """
    def build(self):
        """
        Construct the LSTM with the correct parameters.
        Input is the required states.
        """
        pass
class LSTMs(LSTM):
    """General class to construct Long Short Term Memory RNN."""
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 model='LSTM1',
                 **kwargs):
        """
        Construct the LSTM class.
        Perameters are differnt types of gates that can be reduced based on how Slim the LSTM is.
        """
        if implementation == 0:
            warnings.warn('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if K.backend() == 'theano':
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.
        if model == 'LSTM1':
            cell = LSTMCell1(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model =='LSTM3':
            cell = LSTMCell3(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        elif model == 'LSTM4':
            cell = LSTMCell4(units,
                        activation=activation,
                        recurrent_activation=recurrent_activation,
                        use_bias=use_bias,
                        kernel_initializer=kernel_initializer,
                        recurrent_initializer=recurrent_initializer,
                        unit_forget_bias=unit_forget_bias,
                        bias_initializer=bias_initializer,
                        kernel_regularizer=kernel_regularizer,
                        recurrent_regularizer=recurrent_regularizer,
                        bias_regularizer=bias_regularizer,
                        kernel_constraint=kernel_constraint,
                        recurrent_constraint=recurrent_constraint,
                        bias_constraint=bias_constraint,
                        dropout=dropout,
                        recurrent_dropout=recurrent_dropout,
                        implementation=implementation)
        super(LSTM, self).__init__(cell,             # super of lstm not lstms
                                   return_sequences=return_sequences,
                                   return_state=return_state,
                                   go_backwards=go_backwards,
                                   stateful=stateful,
                                   unroll=unroll,
                                   **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

def _generate_dropout_ones(inputs, dims):
    # Currently, CNTK can't instantiate `ones` with symbolic shapes.
    # Will update workaround once CNTK supports it.
    if K.backend() == 'cntk':
        ones = K.ones_like(K.reshape(inputs[:, 0], (-1, 1)))
        return K.tile(ones, (1, dims))
    else:
        return K.ones((K.shape(inputs)[0], dims))


def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)
################################################################
def read_data(file):
    """
    Read simulation data.
    Data will likely be a csv formatted file.
    """
    data = pd.read_csv(file)

    return data


def clean_data(file):
    """
    Process the data.
    Data cleaning may include reduction or augmentation depending on how skewed the data is.
    """
    df = read_data(file).drop_duplicates( subset = ['TITLE'] )
    dt = df[['CATEGORY', 'TITLE']].sample(50000, random_state = 22)
    cat_count = pd.DataFrame(dt.CATEGORY.value_counts())
    return cat_count
def get_keras_model(num_hidden_layers,
                    num_neurons_per_layer,
                    dropout_rate,
                    activation,
                    optimizer,
                    learning_rate,
                    batch_size,
                    epochs):
    """
    Construct the RNN model with the hyperparameters we want.
    Hyperparameters may include more than what is provided.
    """
    seed = 22
    np.random.seed(seed)
    model = Sequential()
    #model.add(Embedding(10000, 64,input_length = X_train.shape[1] ))
    model.add(SpatialDropout1D(0.7 ))
    model.add(Conv1D(64,5,padding='valid',activation='relu',strides=1))
    model.add(MaxPooling1D(pool_size=2 ))
    #model.add(LSTM(64, activation ='relu'))#, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dropout(0.5 ))
    model.add(Dense(4,activation='softmax'))
    adam = Adam(lr = 0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 )
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    return 0

#set hyperparameter space
p = {'lr': (0.5, 5, 10),
     'first_neuron':[4, 8, 16, 32, 64],
     'hidden_layers':[0, 1, 2],
     'batch_size': (2, 30, 10),
     'epochs': [150],
     'dropout': (0, 0.5, 5),
     'optimizer': ['Adam', 'Nadam', 'RMSprop'],
     'activation':['relu', 'elu']}




#visualize results
#running a grid search across different parameters
batch = [250, 500 , 1000]
eta = [0.0001, 0.00025,  0.0005]
act = ['sigmoid', 'relu' , 'tanh' ]
lstms = ['LSTM' , 'LSTM3', 'LSTM4']
epoch = [15, 20, 30]
dropout_rate = [0.1, 0.2, 0.3]


for bat in batch:
    for i in eta:
        for j in act:
            for k in lstms:
                for p in epoch:
                    for d in dropout_rate:
                        pass

cat_count = clean_data('uci-news-aggregator.csv')
#plot the catgeories
plt.bar(cat_count.index,cat_count['CATEGORY'], color = ['g','b','c','y'])


ax = cat_count['CATEGORY'].plot(kind='bar' ,color = ['g','b','c','y'],figsize=(5,5), edgecolor=None)

for p in ax.patches[:4]:
    width, height = p.get_width(), p.get_height()
    perc = height/cat_count['CATEGORY'].sum()
    x, y = p.get_xy()
    ax.annotate('{:.0%}'.format(perc), (x+0.3, y + height + 70))


plt.xticks(cat_count.index,['Entertainment' , 'Buisiness', 'Technology','Health'],
           rotation=45, fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Categories' ,  fontsize=20)
plt.title('Count of News Articles in each Category',   fontsize=20)
