import logging
import math
import random
import threading

random.seed(30)

Nx = 10

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def input_number_to_vector(number):
    result = [-0.5] * Nx
    set_input_number_to_vector(number,result)
    return result

def set_input_number_to_vector(number, result):
    for i in range(Nx):
        result[i] = -0.5
    floor = int(number)
    lower_part = number - floor
    upper_part = 1 - lower_part
    result[floor] = result[floor] + upper_part
    result[floor + 1] = result[floor + 1] + lower_part

def input_number_to_expected_result(number):
    result = [None] * Nx
    # result = [0.0] * Nx
    set_input_number_to_expected_result(number,result)
    return result

def set_input_number_to_expected_result(number, result):
    for i in range(Nx):
        result[i] = -0.5
    index = round(number)
    result[index] = result[index] + 1.0

def accumulated_error(sample, estimate):
    accum = 0.0
    for i in range(Nx):
        accum = accum + abs(sample[i] - estimate[i])
    return accum

def evaluate_network(xi,ws,bs, yh):
    for i in range(Nx):
        yh[i] = bs[i]
        for j in range(Nx):
            yh[i] = yh[i] + ws[i][j] * xi[i]


def main():
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")
    ws = [[0.0]*Nx] * Nx
    bs = [0.0] * Nx
    for i in range(Nx):
        for j in range(Nx):
            ws[i][j] = random.uniform(-0.5,0.5)
        bs[i] = random.uniform(-0.5,0.5)

    #print(ws)
    #print(bs)
    LEARNING_PASSES = 2000
    M=100
    LC = 0.001
    NUM_THREADS = 12
    print("threads = " + str(NUM_THREADS))
    for p in range(LEARNING_PASSES):
        if p % 100 == 0:
            logging.info("Starting pass " + str(p))

        dws_per_thread = [[None ,None]] * NUM_THREADS
        if (NUM_THREADS > 1):
            threads = []
            for thread_index in range(NUM_THREADS):
                range_start = int(thread_index * M / NUM_THREADS)
                range_end = int((thread_index + 1) * M / NUM_THREADS)
                threads.append(threading.Thread(target=calculate_proposed_wb_changes, args=(range_start, range_end, ws, bs, M, dws_per_thread, thread_index)))
                threads[thread_index].start()

            for thread_index in range(NUM_THREADS):
                threads[thread_index].join()
        else:
            calculate_proposed_wb_changes(0, M, ws, bs, M, dws_per_thread, 0)

        #apply learning
        for thread_index in range(NUM_THREADS):
            dws = dws_per_thread[thread_index][0]
            dbs = dws_per_thread[thread_index][1]
            for i in range(Nx):
                bs[i] = bs[i] + (dbs[i] * LC)
                for j in range(Nx):
                    ws[i][j] = ws[i][j] + (dws[i][j] * LC)


    #final test
    NUM_TEST_SAMPLES= 100
    failure_rate = 0.0
    result_vector = [0.0] * Nx
    for i in range(NUM_TEST_SAMPLES):
        random_number = random.uniform(0.0, Nx - 1.0000000000001)
        xi = input_number_to_vector(random_number)
        evaluate_network(xi,ws,bs, result_vector)

        max = -1000.0
        max_index = -1
        for j in range(Nx):
            if result_vector[j] > max:
                max_index = j
                max = result_vector[j]

        incorrect = ""
        if max_index != round(random_number):
            incorrect = " INCORRECT!"
            failure_rate = failure_rate + 1.0/NUM_TEST_SAMPLES
        print("Rounded " + str(random_number) + " to " + str(max_index) + incorrect)

    print("FAILURE RATE " + str(failure_rate * 100.0) + " %")


def calculate_proposed_wb_changes(range_start, range_end, ws, bs, M, dws_per_thread, thread_index):
    dws = [[0.0]*Nx]*Nx
    dbs = [0.0] *Nx
    error = 0.0
    xi = input_number_to_vector(0.0)
    expected_result = input_number_to_expected_result(0.0)
    yh = [0.0] * Nx
    for m in range(range_start, range_end):
        random_number = random.uniform(0.0, Nx - 1.0000000000001)
        set_input_number_to_vector(random_number, xi)
        set_input_number_to_expected_result(random_number, expected_result)

        # evaluate
        evaluate_network(xi, ws, bs, yh)

        # learn
        error = error + accumulated_error(expected_result, yh) / M
        L = [0.0] * Nx
        for i in range(Nx):
            L[i] = expected_result[i] - yh[i]
            for j in range(Nx):
                dws[i][j] = dws[i][j] + ((xi[i] * L[i]) / M)
            dbs[i] = dbs[i] + (L[i] / M)

    dws_per_thread[thread_index][0] = dws
    dws_per_thread[thread_index][1] = dbs


if __name__ == '__main__':
    main()

