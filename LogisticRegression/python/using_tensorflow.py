import math
import numpy as np
import tensorflow as tf
import random
import time


random.seed(30)

Nx = 100

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def set_input_number_to_vector(number, result, m):
    floor = int(number)
    lower_part = number - floor
    upper_part = 1 - lower_part
    result[floor,m] = result[floor,m] + upper_part
    result[floor + 1,m] = result[floor + 1,m] + lower_part

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
    LEARNING_PASSES = 5000
    M = 1000
    LC = 2.

    X = tf.placeholder(tf.float32, shape=(Nx, M))
    Y = tf.placeholder(tf.float32, shape=(Nx, M))

    W = tf.get_variable("W", [Nx,Nx], initializer = tf.initializers.random_uniform(minval=-0.01,maxval=0.01,seed = 30))
    B = tf.get_variable("B", [Nx, 1], initializer=tf.zeros_initializer())

    Yhat = tf.add(tf.matmul(W, X), B)
    error_value = tf.reduce_mean(tf.subtract(Yhat, Y)**2)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LC).minimize(error_value)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for p in range(LEARNING_PASSES):
            X_input = np.zeros((Nx, M))
            Y_input = np.zeros((Nx, M))
            for m in range(M):
                r = random.uniform(0.0, Nx - 1.0000000000001)
                set_input_number_to_vector(r, X_input, m)
                set_input_number_to_expected_result(r, Y_input, m)

            _ , cost = sess.run([optimizer,error_value], feed_dict={ X: X_input, Y: Y_input })

            if p % 200 == 1:
                print("Vectorized error = " + str(cost))
        Wout = sess.run(W)
        Bout = sess.run(B)
    return Wout,Bout

def main():
    start = time.time()
    W, B = train()
    end = time.time()
    print("Elapsed time = %s" % (end - start))

    #final test
    NUM_TEST_SAMPLES = 100
    failure_rate = 0.0
    for i in range(NUM_TEST_SAMPLES):

        random_number = (random.uniform(0.0, Nx - 1.0000000000001))
        X_test = np.zeros((Nx, 1))
        set_input_number_to_vector(random_number,X_test,0)
        result_vector_vectorized = np.dot(W,X_test) + B

        expected_result = round(random_number)
        result_vector = result_vector_vectorized#evaluate_network(xi,ws,bs)

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

