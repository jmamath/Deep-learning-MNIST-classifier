"""
Created on Sat Oct 14 10:27:59 2017

@author: jmamath
"""

import numpy as np
import h5py


def load_dataset():
    dataset = h5py.File('mnist.h5', "r")
    train_x = np.array(dataset["train_x"][:]) # your train set features
    train_y = np.array(dataset["train_y"][:]) # your train set labels

    dev_x = np.array(dataset["dev_x"][:])   # your dev set features
    dev_y = np.array(dataset["dev_y"][:])   # your dev set labels

    test_x = np.array(dataset["test_x"][:])  # your test set features
    test_y = np.array(dataset["test_y"][:])  # your test set labels

    train_y = train_y.reshape((train_y.shape[0],1))
    dev_y = dev_y.reshape((dev_y.shape[0],1))
    test_y = test_y.reshape((test_y.shape[0],1))



    return train_x, train_y, dev_x, dev_y, test_x, test_y

train_x, train_y, dev_x, dev_y, test_x, test_y = load_dataset()

# This function reshaping is meant to transform a vector of
# labeled classes from 0 to 9 into a matrix
# Hence we are transforming a 4 into a [0,0,0,0,1,0,0,0,0,0].reshape(10,1)
#                           a 5 into a [0,0,0,0,0,1,0,0,0,0].reshape(10,1)
#                           a 6 into a [0,0,0,0,0,0,1,0,0,0].reshape(10,1)
#                           etc.
def reshaping(vect_y):
    vect_y = vect_y + 1
    One = np.ones(10).reshape(1,10)
    Diag = np.diag([1./i for i in range(1,11)])
    vect_y = np.dot(np.dot(vect_y,One),Diag)
    vect_y = (vect_y * (vect_y == 1))
    return vect_y



def load_preprocessed_dataset():
    """
    Returns:
    train_x -- train set of features, shape (784, 50000)
    train_y -- train set of labels, shape (10, 50000)
    test_x -- test set of features, of shape (784, 10000)
    test_y -- test set of labels, shape (10, 10000)
    """

    train_x, train_y, dev_x, dev_y, test_x, test_y = load_dataset()

    train_y = reshaping(train_y)
    test_y = reshaping(test_y)

    train_x = train_x.T
    train_y = train_y.T
    test_x = test_x.T
    test_y = test_y.T

    return train_x,train_y,test_x,test_y
