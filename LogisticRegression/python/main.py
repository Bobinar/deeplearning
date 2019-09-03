import math
import random

random.seed(30)

Nx = 100

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def input_number_to_vector(number):
    result = [0.0] * Nx
    floor = int(number)
    lower_part = number - floor
    upper_part = 1 - lower_part
    result[floor] = result[floor] + upper_part
    result[floor + 1] = result[floor + 1] + lower_part
    return result

def input_number_to_expected_result(number):
    result = [0.0] * Nx
    index = round(number)
    result[index] = result[index] + 1.0
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

def main():

    ws = [0.0]*Nx
    bs = [0.0] * Nx
    for i in range(Nx):
        ws[i] = [0.0] *Nx
        for j in range(Nx):
            ws[i][j] = random.uniform(-0.5,0.5)
        bs[i] = random.uniform(-0.5,0.5)

    LEARNING_PASSES = 5000
    M=1000
    LC = 0.01

    for p in range(LEARNING_PASSES):

        dws = [0.0]*Nx
        for i in range(Nx):
            dws[i] = [0.0] *Nx
        dbs = [0.0] * Nx
        error = 0.0
        for m in range(M):
            random_number = random.uniform(0.0,Nx - 1.0000000000001)
            xi = input_number_to_vector(random_number)
            expected_result = input_number_to_expected_result(random_number)

            # evaluate
            yh = evaluate_network(xi,ws,bs)

            #learn
            error = error + accumulated_error(expected_result,yh) /M
            L = [0.0] * Nx
            for i in range(Nx):
                error_i = yh[i] - expected_result[i]
                for j in range(Nx):
                    dws[i][j] = dws[i][j] + ((xi[j] * error_i)/M)
                dbs[i] = dbs[i] + (error_i / M)

        if p % 100 == 1:
            print(error)

        #apply learning
        for i in range(Nx):
            bs[i] = bs[i] - (dbs[i] * LC)
            for j in range(Nx):
                ws[i][j] = ws[i][j] - (dws[i][j] * LC)

    #final test
    NUM_TEST_SAMPLES = 100
    failure_rate = 0.0
    for i in range(NUM_TEST_SAMPLES):

        random_number = (random.uniform(0.0, Nx - 1.0000000000001))
        xi = input_number_to_vector(random_number)
        expected_result = round(random_number)
        result_vector = evaluate_network(xi,ws,bs)

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

