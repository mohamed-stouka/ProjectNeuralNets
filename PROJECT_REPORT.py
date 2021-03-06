"""This file contains the stub code of CMSE802 Project.

It contains classes of slim LSTM (reduced parameters) which will be used to evaluate the computed hyperparameters.
"""
import warnings
from keras.layers import LSTM
from keras import backend as K
from keras import regularizers
from keras.layers import LSTMCell


class LSTMCell1(LSTMCell):
    """Original LSTM with non reduced parameters."""
    
    def build() :
        """
        Construct the LSTM with the correct parameters.
        
        Input is the required states.
        """
        return 0
       


class LSTMCell3(LSTMCell):
    """
    LSTM with the required reduced parameters.
    
    Each gate is computed using only the bias.
    """
    
    def build():
        """
        Construct the LSTM with the correct parameters.
        
        Input is the required states.
        """
        return 0

        
class LSTMCell4(LSTMCell):
    """
    LSTM with the required reduced parameters.
    
    Each gate is computed using only the previous hidden state but with point wise multiplication.
    """
    
    def build():
        """
        Construct the LSTM with the correct parameters.
        
        Input is the required states.
        """
        return 0
   

    
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
        
        Perameters are differnt types of gates the can be kept or reduced based on how Slim the LSTM needs to be.
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
            return

        elif model =='LSTM3':
            return

        elif model == 'LSTM4':
            return

def read_data(file):
    """
    Read simulation data.
    
    Data will likely be a csv formatted file.
    """
    return 0


def clean_data():
    """
    Process the data.
    
    Data cleaning may include reduction or augmentation depending on how skewed the data is.
    """
    return 0


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


