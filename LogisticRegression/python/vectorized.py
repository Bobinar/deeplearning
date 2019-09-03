import math
import random
import numpy as np
import time
from numba import jit



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
def train():
    LEARNING_PASSES = 5000
    M = 1000
    LC = 0.01
    W = np.random.randn(Nx, Nx) * 0.01
    B = np.zeros((Nx, 1))
    p=0
    for p in range(LEARNING_PASSES):

        X = np.zeros((Nx, M))
        Y = np.zeros((Nx, M))
        for m in range(M):
            r = random.uniform(0.0, Nx - 1.0000000000001)
            set_input_number_to_vector(r, X, m)
            # set_input_number_to_expected_result(r, X, m)
            set_input_number_to_expected_result(r, Y, m)

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
    start = time.time()
    W, B = train()
    end = time.time()
    print("Elapsed (with compilation) = %s" % (end - start))

    start = time.time()
    W, B = train()
    end = time.time()
    print("Elapsed (after compilation) = %s" % (end - start))

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

