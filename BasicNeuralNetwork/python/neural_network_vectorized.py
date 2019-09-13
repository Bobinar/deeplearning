import math
import random
import numpy as np
import time
from numba import jit



random.seed(30)

Nx = 10

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
    LEARNING_PASSES = 50000
    M = 1000
    LC = 0.01
    W1 = np.random.randn(Nx, Nx) * 0.01
    B1 = np.zeros((Nx, 1))
    W2 = np.random.randn(Nx, Nx) * 0.01
    B2 = np.zeros((Nx, 1))

    X = np.zeros((Nx, M))
    Y = np.zeros((Nx, M))
    for m in range(M):
        r = random.uniform(0.0, Nx - 1.0000000000001)
        set_input_number_to_vector(r, X, m)
        set_input_number_to_expected_result(r, Y, m)

    p=0
    for p in range(LEARNING_PASSES):
        A1= np.dot(W1, X) + B1
        Yhat = np.dot(W2,A1)+ B2

        error_vector = Yhat - Y

        dW2 = np.dot(error_vector, A1.T) / M
        dB2 = np.sum(error_vector, axis=1).reshape(B2.shape) / M

        inverse_W2 = np.linalg.inv(W2)

        YminusBias = Y - B2
        A1_desired = np.dot(inverse_W2,YminusBias)
        error_vector_first_layer = A1 - A1_desired
        dW1 = np.dot(error_vector_first_layer, X.T) /M
        dB1 = np.sum(error_vector_first_layer, axis=1).reshape(B1.shape) / M

        W1  = W1 - dW1 * LC
        B1 = B1 - dB1 * LC
        W2 = W2 - dW2 * LC
        B2 = B2 - dB2 * LC

        #vectorized_error = np.sum(np.abs(error_vector), axis=1, keepdims=True) / M
        #vectorized_error = np.sum(vectorized_error)

        #if p % 100 == 1:
        #    print("Vectorized error = " + str(vectorized_error))

    return W1,B1,W2,B2

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
        result_vector_vectorized = np.dot(W2,A1) + B2
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

