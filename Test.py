
"""
Created on Sat Oct 14 10:27:59 2017

@author: jmamath
"""

import numpy as np
import math
import tensorflow as tf
from tensorflow.python.framework import ops
from Load_data import reshaping
from model_utils import *



# reshaping unit test
y = np.array([1,2,0,9]).reshape(4,1)
reshaping(y)

## initialize_parameters_deep unit test
ops.reset_default_graph()
parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

## create_placeholders unit test
X, Y = create_placeholders(5, 3)
print ("X = " + str(X))
print ("Y = " + str(Y))

##
def linear_forward_test_case():
    np.random.seed(1)
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    return A, W, b

## linear_forward and linear activation unit test
A, W, b = linear_forward_test_case()
Z = linear_forward(A, W, b)
A = linear_activation_forward(Z)
print("Z = " + str(Z))
print("A = " + str(A))

## Test of L_model_forward and cost_withRegularization
ops.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(5, 3)
    parameters = initialize_parameters_deep([5,4,3])
    AL = L_model_forward(X,parameters)
    cost = cost_withRegularization(AL, Y, parameters, beta = 0.01)
    print("cost = " + str(cost))
    sess.close()
