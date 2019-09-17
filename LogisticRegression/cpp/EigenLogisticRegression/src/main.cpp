#include <iostream>
#include <Eigen/Dense>
#include <random>
#include <chrono>

const int LEARNING_PASSES = 5000;
const int M = 1000;
const float LC = 0.01f;
const int Nx = 100;

void set_input_number_to_vector(float number, Eigen::MatrixXf &result, int m) {
    int floor = std::floor(number);
    float lower_part = number - floor;
    float upper_part = 1 - lower_part;
    result(floor, m) = result(floor, m) + upper_part;
    result(floor + 1, m) = result(floor + 1, m) + lower_part;
}

void set_input_number_to_expected_result(float number, Eigen::MatrixXf &result, int m) {
    int index = std::round(number);
    result(index, m) = result(index, m) + 1.0f;
}

void train(Eigen::MatrixXf &W, Eigen::VectorXf &B)
{
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> randomDist(0.0f, Nx - 1.001f);

    Eigen::MatrixXf X(Nx,M);
    Eigen::MatrixXf Y(Nx,M);
    Eigen::MatrixXf Yhat(Nx,M);
    Eigen::MatrixXf errorVector(Nx,M);

    Eigen::MatrixXf dW(Nx,Nx);
    Eigen::MatrixXf dB(Nx,1);

    for(int p=0; p < LEARNING_PASSES; p++)
    {
        X = X.Constant(Nx,M,0.0f);
        Y = Y.Constant(Nx,M,0.0f);
        for (int m = 0; m < M; m++)
        {
            float randomNumber = randomDist(e2);
            set_input_number_to_vector(randomNumber,X,m);
            set_input_number_to_expected_result(randomNumber,Y,m);
        }

        Yhat = W * X;
        Yhat.colwise() += B;
        errorVector = Yhat - Y;

        dW = (errorVector * X.transpose()) / M;
        dB = errorVector.rowwise().sum() / M;

        W = W - (dW * LC);
        B = B - (dB * LC);
        /*
        if(p % 100 == 1) {

            float vectorized_error = errorVector.norm()/M;
            std::cout << vectorized_error << std::endl;
        }*/
    }
}



int main()
{
    std::cout << "nbThreads " << Eigen::nbThreads() << std::endl;

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> randomDist(0.0f, Nx - 1.001f);

    Eigen::MatrixXf W(Nx,Nx);
    W = W.Random(Nx,Nx) * 0.01f;
    Eigen::VectorXf B(Nx);
    B = Eigen::VectorXf::Constant(Nx,0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    train(W,B);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    const int NUM_TEST_SAMPLES = 100;
    float failure_rate = 0.0;
    for (int i=0;i< NUM_TEST_SAMPLES; i++)
    {
        Eigen::MatrixXf X(Nx,1);
        Eigen::MatrixXf Yhat(Nx,1);

        X = X.Constant(Nx,1,0.0f);

        float randomNumber = randomDist(e2);
        set_input_number_to_vector(randomNumber,X,0);

        Yhat = (W*X) + B;
        float max = -1000.0;
        int max_index = -1;
        for (int j=0;j<Nx;j++){
            if (Yhat(j,0) > max) {
                max_index = j;
                max = Yhat(j,0);
            }
        }

        int expectedResult = std::round(randomNumber);

        std::string incorrect;
        if (max_index != expectedResult) {
            incorrect = " INCORRECT!";
            failure_rate += 1.0f/NUM_TEST_SAMPLES;
        }
        std::cout << "Rounded " << randomNumber << " to " << max_index <<  incorrect << std::endl;
    }
    std::cout << "FAILURE RATE " << (failure_rate * 100.0) << " %" << std::endl;

	return 0;
}