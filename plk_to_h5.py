"""
Created on Tue Sep 26 12:10:09 2017

This code is meant to be execute in python 2

@author: jmamath
"""
import gzip
import pickle
import numpy as np
import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt


with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_x, train_y = train_set
dev_x,dev_y = valid_set
test_x, test_y = test_set

## Creating a h5 file to store mnist data
## We will have 6 different dataset to play with
hf = h5py.File('mnist.h5', 'w')

hf.create_dataset('train_x', data=train_x)
hf.create_dataset('train_y', data=train_y)

hf.create_dataset('dev_x', data=dev_x)
hf.create_dataset('dev_y', data=dev_y)

hf.create_dataset('test_x', data=test_x)
hf.create_dataset('test_y', data=test_y)

hf.close()


#Display one digit
plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()
