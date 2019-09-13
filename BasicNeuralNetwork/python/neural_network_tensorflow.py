import math
import numpy as np
import tensorflow as tf
import random
import time
from numba import jit

random.seed(30)

Nx = 10

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

@jit
def create_random_examples(M):
    X = np.zeros((Nx, M))
    Y = np.zeros((Nx, M))
    for m in range(M):
        r = random.uniform(0.0, Nx - 1.0000000000001)
        set_input_number_to_vector(r, X, m)
        set_input_number_to_expected_result(r, Y, m)
    return X, Y

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


def train():
    LEARNING_PASSES = 10000
    M = 10000
    LC = 0.01


    X = tf.placeholder(tf.float32, shape=(Nx, M))
    Y = tf.placeholder(tf.float32, shape=(Nx, M))

    W1 = tf.get_variable("W1", [Nx,Nx], initializer = tf.initializers.random_uniform(minval=-0.01,maxval=0.01,seed = 30))
    B1 = tf.get_variable("B1", [Nx, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [Nx,Nx], initializer = tf.initializers.random_uniform(minval=-0.01,maxval=0.01,seed = 20))
    B2 = tf.get_variable("B2", [Nx, 1], initializer=tf.zeros_initializer())

    A1 = tf.nn.relu(tf.add(tf.matmul(W1, X), B1))
    Yhat = tf.sigmoid(tf.add(tf.matmul(W2,A1),B2))
    error_value = tf.reduce_mean(tf.subtract(Yhat, Y)**2)
    optimizer = tf.train.AdamOptimizer(learning_rate=LC).minimize(error_value)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for p in range(LEARNING_PASSES):

            X_input, Y_input = create_random_examples(M)
            _ , cost = sess.run([optimizer,error_value], feed_dict={ X: X_input, Y: Y_input })

            if p % 200 == 1:
                print("Vectorized error = " + str(cost))
        W1out = sess.run(W1)
        B1out = sess.run(B1)
        W2out = sess.run(W2)
        B2out = sess.run(B2)
    return W1out,B1out,W2out,B2out

def main():
    start = time.time()
    W1, B1, W2, B2 = train()
    end = time.time()
    print("Elapsed time = %s" % (end - start))

    #final test
    NUM_TEST_SAMPLES = 100
    failure_rate = 0.0
    for i in range(NUM_TEST_SAMPLES):

        random_number = (random.uniform(0.0, Nx - 1.0000000000001))
        X_test = np.zeros((Nx, 1))
        set_input_number_to_vector(random_number,X_test,0)

        A1 = np.dot(W1,X_test) + B1
        result_vector = np.dot(W2,A1) + B2
        expected_result = round(random_number)

        max = -1000.0
        max_index = -1
        for j in range(Nx):
            if result_vector[j] > max:
                max_index = j
                max = result_vector[j]

        incorrect = ""
        if max_index != expected_result:
            incorrect = " INCORRECT!"
            failure_rate = failure_rate + 1.0/NUM_TEST_SAMPLES
        print("Rounded " + str(random_number) + " to " + str(max_index) + incorrect)
    print("FAILURE RATE " + str(failure_rate * 100.0) + " %")

if __name__ == '__main__':
    main()

