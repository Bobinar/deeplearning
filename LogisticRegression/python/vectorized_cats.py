import math
import random
import numpy as np
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

Nx = 100

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

@jit
def set_input_number_to_vector(number, result, m):
    floor = int(number)
    lower_part = number - floor
    upper_part = 1 - lower_part
    result[floor,m] = result[floor,m] + upper_part
    result[floor + 1,m] = result[floor + 1,m] + lower_part

@jit
def set_input_number_to_expected_result(number, result, m):
    index = round(number)
    result[index,m] = result[index,m] + 1.0
    return result

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

@jit
def train(X, Y):
    LEARNING_PASSES = 5000
    M = X.shape[1]
    LC = 0.01
    W = np.random.randn(1, X.shape[0]) * 0.01
    B = np.zeros((1, 1))
    p=0
    for p in range(LEARNING_PASSES):
        Yhat = np.dot(W, X) + B

        error_vector = Yhat - Y

        dW = np.dot(error_vector, X.T) / M
        #dB = np.sum(error_vector, axis=1, keepdims=True) / M
        dB = np.sum(error_vector, axis=1).reshape(B.shape) / M
        W = W - dW * LC
        B = B - dB * LC

        #vectorized_error = np.sum(np.abs(error_vector), axis=1, keepdims=True) / M
        #vectorized_error = np.sum(vectorized_error)

        #if p % 100 == 1:
        #    print("Vectorized error = " + str(vectorized_error))

    return W,B

def main():

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255.
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T /255.

    print(test_set_x_flatten.shape)

    start = time.time()
    W, B = train(train_set_x_flatten, train_set_y_orig)
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))
    '''
    start = time.time()
    W, B = train(train_set_x_flatten, train_set_y_orig)
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))
    '''
    #final test
    NUM_TEST_SAMPLES = test_set_x_flatten.shape[1]
    failure_rate = 0.0

    Yhat = np.dot(W, test_set_x_flatten) + B

    for i in range(NUM_TEST_SAMPLES):
        Yhat_sample = Yhat[0,i]
        if Yhat_sample > 0.5:
            Yhat_sample = 1
        else:
            Yhat_sample = 0


        incorrect = ""
        if Yhat_sample != test_set_y_orig[0,i]:
            incorrect = " INCORRECT!"
            failure_rate = failure_rate + 1.0/NUM_TEST_SAMPLES
        print("image " + str(i) + incorrect)
    print("FAILURE RATE " + str(failure_rate * 100.0) + " %")

if __name__ == '__main__':
    main()

