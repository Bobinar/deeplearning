import math
import numpy as np
import tensorflow as tf
import random
import time
from numba import jit

import h5py

def load_dataset():
    train_dataset = h5py.File('../../datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('../../datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


random.seed(30)

def relu(X):
    return np.fmax(X,0)

def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def accumulated_error(sample, estimate):
    accum = 0.0
    for i in range(Nx):
        accum = accum + abs(sample[i] - estimate[i])
    return accum

def evaluate_network(xi,ws,bs):
    yh = [0.0] * Nx

    for i in range(Nx):
        yh[i] = bs[i]
        for j in range(Nx):
            yh[i] = yh[i] + ws[i][j] * xi[j]
    return yh


def train(X_input,Y_input):
    LEARNING_PASSES = 1000
    M = X_input.shape[1]
    LC = 0.001

    Nx = X_input.shape[0]

    HIDDEN_LAYER_SIZE = 10

    X = tf.placeholder(tf.float32, shape=X_input.shape)
    Y = tf.placeholder(tf.float32, shape=Y_input.shape)

    W1 = tf.get_variable("W1", [HIDDEN_LAYER_SIZE,Nx], initializer = tf.initializers.random_uniform(minval=-0.01,maxval=0.01,seed = 30))
    B1 = tf.get_variable("B1", [HIDDEN_LAYER_SIZE, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [1,HIDDEN_LAYER_SIZE], initializer = tf.initializers.random_uniform(minval=-0.01,maxval=0.01,seed = 20))
    B2 = tf.get_variable("B2", [1, 1], initializer=tf.zeros_initializer())

    A1 = tf.nn.relu(tf.add(tf.matmul(W1, X), B1))
    Yhat = tf.sigmoid(tf.add(tf.matmul(W2,A1),B2))
    error_value = tf.reduce_mean(tf.subtract(Yhat, Y)**2)
    optimizer = tf.train.AdamOptimizer(learning_rate=LC).minimize(error_value)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for p in range(LEARNING_PASSES):


            _ , cost = sess.run([optimizer,error_value], feed_dict={ X: X_input, Y: Y_input })

            if p % 200 == 1:
                print("Vectorized error = " + str(cost))
        W1out = sess.run(W1)
        B1out = sess.run(B1)
        W2out = sess.run(W2)
        B2out = sess.run(B2)
    return W1out,B1out,W2out,B2out

def test_on_dataset(X,Y,W1,B1,W2,B2):
    NUM_TEST_SAMPLES = X.shape[1]
    A1 = relu(np.dot(W1, X) + B1)
    Yhat = sigmoid(np.dot(W2, A1) + B2)
    failure_rate = 0.0
    for i in range(NUM_TEST_SAMPLES):
        Yhat_sample = Yhat[0, i]
        if Yhat_sample > 0.5:
            Yhat_sample = 1
        else:
            Yhat_sample = 0

        incorrect = ""
        if Yhat_sample != Y[0, i]:
            incorrect = " INCORRECT!"
            failure_rate = failure_rate + 1.0 / NUM_TEST_SAMPLES
        #print("image " + str(i) + incorrect)
    print("FAILURE RATE " + str(failure_rate * 100.0) + " %")

def main():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    train_set_x_flatten = (train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255.) -0.5
    test_set_x_flatten = (test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T /255.) - 0.5

    start = time.time()
    W1, B1, W2, B2 = train(train_set_x_flatten, train_set_y_orig)
    end = time.time()
    print("Elapsed time = %s" % (end - start))

    print("Testing against train set")
    test_on_dataset(train_set_x_flatten, train_set_y_orig, W1, B1, W2, B2)

    print("Testing against test set")
    test_on_dataset(test_set_x_flatten, test_set_y_orig, W1, B1, W2, B2)
    #


if __name__ == '__main__':
    main()

