"""
Created on Sat Oct 14 10:27:59 2017

@author: jmamath
"""

############ 1 - Packages ############

import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.python.framework import ops
import time
from model_utils import *
from Load_data_v import load_preprocessed_dataset

############ 2 - Dataset ############

train_x,train_y,test_x,test_y = load_preprocessed_dataset()

#Display one digit
index = 56
plt.imshow(train_x.T[index].reshape((28, 28)))

# Explore your dataset
m_train = train_x.shape[1]
m_test = test_x.shape[1]
n_y = train_y.shape[0]
n_x = train_x.shape[0]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(28))
print ("number of classes: n_y = " + str(n_y))

print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))

############ 3 - L-layer neural network ############

def L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, num_epochs = 1500, minibatch_size = 32, learning_rate = 0.001, beta = 0.01):

    """
    Implements a L-layer neural network: LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- input data, of shape (n_x, number of examples)
    Y_train -- test labels represented by a numpy array (vector) of shape (n_y, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    beta -- hyperparameter for regularization.
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch

    """
    tic = time.time()
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    costs = []                              # to keep track of the cost
    m = X_train.shape[1]                           # number of examples
    n_x,n_y = layers_dims[0],layers_dims[len(layers_dims)-1]
    X,Y = create_placeholders(n_x, n_y)

    ## Initializing parameters
    parameters = initialize_parameters_deep(layers_dims)

    # Forward propagation
    AL = L_model_forward(X, parameters)

    ## Computing cost
    cost = cost_withRegularization(AL, Y, parameters, beta)

    # Backpropagation: Define the tensorflow optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess :
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
           # Print the cost every epoch
            if epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            costs.append(epoch_cost)
        # Ploting cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))

        # Calculate accuracy on the training and test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("Train Accuracy: %", 100*accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy: %", 100*accuracy.eval({X: X_test, Y: Y_test}))
        sess.close()
    toc = time.time()
    print("Time :" + str(math.floor(toc-tic)) + "s")


h_1 = 200 # No neurons in the 1st hidden layer
h_2 = 100 # No neurons in the 2nd hidden layer
h_3 = 50  # No neurons in the 3rd hidden layer
layers_dims = [n_x,h_1,h_2,h_3, n_y]

L_layer_model(train_x, train_y, test_x, test_y,layers_dims,num_epochs = 12, minibatch_size = 1024, learning_rate = 0.001, beta = 0.001)   
# 64 epoch is the gold
# I get 99.84% on the training set
# and 98.05% on the test set for 390s
