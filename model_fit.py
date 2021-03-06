# -*- coding: utf-8 -*-
import create_keras_model as ckm
import clean_data as cd
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import backend as K
import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from skopt.plots import plot_objective
from skopt.utils import use_named_args


#2019 uci news dataset 
dataset = 'uci-news-aggregator.csv'
cat_count, dt = cd.clean_data(dataset)
X_train, X_test, Y_train, Y_test = ckm.tokenize_text(dt)

#Set the limits for the Hyperparameters
dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform', name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=4, high=128, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'], name='activation')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation]

#Initilize arbitrary the Hyperparameters
default_parameters = [1e-5, 1, 4, 'sigmoid']

#define global variable to store accuracy
best_accuracy = 0.0
validation_data = (X_test, Y_test)

# This is a function to log traning progress so that can be viewed by TnesorBoard.
def log_dir_name(learning_rate, num_dense_layers, 
                 num_dense_nodes, activation):
    """
    Function to log traning progress so that can be viewed by TnesorBoard.
    """
    # The dir-name for the TensorBoard log-dir.
    s = "./19_logs/lr_{0:.0e}_layers_{1}_nodes_{2}_{3}/"
    # Insert all the hyper-parameters in the dir-name.
    log_dir = s.format(learning_rate,
                       num_dense_layers,
                       num_dense_nodes,
                       activation)
    return log_dir

def fitness(param_list):
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:  Number of dense neurons.
    activation:        Activation function for all layers.
    """
    learning_rate = param_list[0]
    num_dense_layers = param_list[1]
    num_dense_nodes = param_list[2]
    activation = param_list[3]
    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(learning_rate))
    print('num_dense_layers:', num_dense_layers)
    print('num_dense_nodes:', num_dense_nodes)
    print('activation:', activation)
    print()
    # Create the neural network with these hyper-parameters.
    model = ckm.create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)
    # Dir-name for the TensorBoard log-files.
    log_dir = log_dir_name(learning_rate, num_dense_layers,
                           num_dense_nodes, activation)
    # Create a callback-function for Keras which will be
    # run after each epoch has ended during training.
    # This saves the log-files for TensorBoard.
    # Note that there are complications when histogram_freq=1.
    # It might give strange errors and it also does not properly
    # support Keras data-generators for the validation-set.
    callback_log = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)
    # Use Keras to train the model.
    history = model.fit(x= X_train,
                        y= Y_train,
                        epochs=3,
                        batch_size=128,
                        validation_data=validation_data,
                        callbacks=[callback_log])
    # Get the classification accuracy on the validation-set
    accuracy = history.history['val_accuracy'][-1]
    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()
    # We use the global keyword so we update the variable outside
    # of this function.
    global best_accuracy

    # If the classification accuracy of the saved model is improved ...
    if accuracy > best_accuracy:
        # Update the classification accuracy.
        best_accuracy = accuracy
    # Delete the Keras model with these hyper-parameters from memory.
    del model
    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    return -accuracy

def minimize():
    """
    Minimize the hyperparameters.
    Grab the lowest paramters for the best accuracy.
    """
    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=default_parameters)
    return search_result

def plot_accuracy(param_list):
    
    """
    Hyper-parameters:
    learning_rate:     Learning-rate for the optimizer.
    num_dense_layers:  Number of dense layers.
    num_dense_nodes:   Number of dense neurons.
    activation:        Activation function for all layers.
    Plots the accuracy of the model oover the training epochs.
    """
    learning_rate = param_list[0]
    num_dense_layers = param_list[1]
    num_dense_nodes = param_list[2]
    activation = param_list[3]
    # Create the neural network with these hyper-parameters.
    model = ckm.create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation)
    # Use Keras to train the model.
    history = model.fit(x= X_train,
                        y= Y_train,
                        epochs=20,
                        batch_size=128,
                        validation_data=validation_data)
    return history
