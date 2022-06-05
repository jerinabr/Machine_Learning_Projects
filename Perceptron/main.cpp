#include <iostream>
#include <fstream>
#include <chrono>
#include <time.h>
#include "../MNISTReader/mnistreader.hpp"

using namespace std;
using namespace chrono;

const double ALPHA = 0.02; // Learning rate
int trainingPasses = 100; // Number of training passes
int testNums[2] = {2, 6}; // Numbers to classify
bool writeToFile = false; // Whether or not to write training weights to a csv file

// Read MNIST dataset
mnistreader mnist(
    "../MNISTDataset/trainImages",
    "../MNISTDataset/trainLabels",
    "../MNISTDataset/testImages",
    "../MNISTDataset/testLabels",
    2000, // Number of training data to use
    200, // Number of testing data to use
    true
);

// Define weights and bias
double* w = (double*) malloc(mnist.imgSize * sizeof(double));
double b;

// Function used to compute the dot product of the weight matrix and the pixel data
double forwardPass(double *x) {
    double sum;
    for (int i = 0; i < mnist.imgSize; ++i) {
        sum += w[i] * x[i];
    }
    sum += b;
    return sum;
}

// Function used to train the perceptron
void train() {
    for (int i = 0; i < mnist.trainSize; ++i) {
        double pred = forwardPass(mnist.trainImgData[i]) * mnist.trainLblData[i];
        if (pred <= 0) {
            for (int j = 0; j < mnist.imgSize; ++j) {
                w[j] += ALPHA * mnist.trainLblData[i] * mnist.trainImgData[i][j];
            }
            b += ALPHA * mnist.trainLblData[i];
        }
    }
}

int main() {
    // Initialize weights and bias
    srand(time(NULL));
    for (int i = 0; i < mnist.imgSize; ++i) {
        w[i] = (double) rand() / RAND_MAX;
    }
    b = (double) rand() / RAND_MAX;

    // Preprocess data
    mnist.selectData(testNums, 2);
    for (int i = 0; i < mnist.trainSize; ++i) {
        if (mnist.trainLblData[i] == testNums[0]) {
            mnist.trainLblData[i] = -1;
        }
        else {
            mnist.trainLblData[i] = 1;
        }
    }

    // Train model
    steady_clock::time_point t0 = steady_clock::now();
    if (writeToFile) {
        ofstream wData;
        wData.open("wData.csv");
        for (int i = 0; i < trainingPasses; ++i) {
            train();
            for (int j = 0; j < mnist.imgSize; ++j) {
                wData << w[j] << ",";
            }
            wData << b << endl;
        }
        wData.close();
    }
    else {
        for (int i = 0; i < trainingPasses; ++i) {
            train();
        }
    }
    steady_clock::time_point t1 = steady_clock::now();
    cout << "Training took " << duration_cast<milliseconds>(t1 - t0).count() << " ms" << endl;

    // Test model
    int truePos = 0;
    int trueNeg = 0;
    int falsePos = 0;
    int falseNeg = 0;
    double accuracy = 0;
    for (int i = 0; i < mnist.testSize; ++i) {
        double pred = forwardPass(mnist.testImgData[i]);
        int actual = mnist.testLblData[i];

        cout << "pred: " << (pred < 0 ? testNums[0] : testNums[1])
        << " | actual: " << actual << endl; // This print statement changes the accuracy lol

        if (actual == testNums[0]) {
            if (pred < 0) {
                trueNeg++;
            }
            else {
                falsePos++;
            }
        }
        else {
            if (pred < 0) {
                falseNeg++;
            }
            else {
                truePos++;
            }
        }
    }
    accuracy = 100.0 * (truePos + trueNeg) / mnist.testSize;
    cout << truePos + trueNeg << "/" << mnist.testSize << " correct" << endl;
    cout << "The model is " << accuracy << "%" << " accurate" << endl;

    return 0;
}