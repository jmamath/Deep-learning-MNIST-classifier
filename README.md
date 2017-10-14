# Deep-learning-MNIST-classifier

## Introduction

When you go at your local post office to send a letter, you usually write the post code manually. 
There is learning algorithms that are able to automatically "read" your written digits.
Our goal is to train such a model using a fully connected neural network and get 98 % accuracy. 
I guess it is possible to reach 99.9% using convolutional neural network, but I haven't learnt it yet.

The MNIST dataset has been used for a while in computer vision, it consist of
hand written digits. More informations
can be found on http://yann.lecun.com/exdb/mnist.

In this project, we are going to use tensorflow (a python package for deep learning).

## Data
You can download the data at the following adress : http://deeplearning.net/data/mnist/mnist.pkl.gz
and launch the script plk_to_h5.py to get a .h5 file.
* train_x is my training set of 50000 hand written digits
* train_y is my label set of 50000 numerical digits corresponding to train_x

* test_x is the test set of features 10000 hand written digits
* test_y is the test set of labels 10000 of digits


## Background

We will use Python and several libraries such as numpy, tensorflow.
A general knowledge of machine learning is expected, and more specifically
deep learning techniques will be used. 

For general information there's lots of good resources on the web. I've personally learnt
all of the general concepts as well as the programming techniques from Pr Andrew Ng
on Coursera  : 
* Machine learning course https://www.coursera.org/learn/machine-learning
* Deep learning specialization https://www.coursera.org/specializations/deep-learning 
Otherwise their is a really exciting video from Siraj Raval
* Intro to deep learning https://youtu.be/vOppzHpvTiQ.
This web site is also worth considering http://deeplearning.net



## The code

There is five files in this project
* plk_to_h5.py unizip and transform the data to hdf5 format.
* Load_data_v.py is the code use to get our data in a relevant format for machine learning.
* model_utils.py is composed of all the relevant functions needed for Model to
works well.
* Test.py contains units test for each functions of utils.
* model.py contains the actual model that we use to train our model.

### Set up note
plk_to_h5 is meant to be executed using python 2.7, I am using a mac, so the program is native in my environment.
All the other programs have been written with python 3.6, I have personally downloaded the Anaconda suite to get all the relevant packages for machine learning and scientific computing.
I created a virtual environment to instal tensorflow.


