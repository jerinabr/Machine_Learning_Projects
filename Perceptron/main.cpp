#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <time.h>
#include "../MNISTReader/mnistreader.hpp"

using namespace std;

const double ALPHA = 0.02; // Learning rate
int trainingPasses = 20; // Number of training passes
vector<int> testNums = {2, 6}; // Numbers to classify
bool writeToFile = false; // Whether or not to write training weights to a csv file

// Read MNIST dataset
mnistreader mnist(
    "../MNISTDataset/trainImages",
    "../MNISTDataset/trainLabels",
    "../MNISTDataset/testImages",
    "../MNISTDataset/testLabels",
    4000, // Number of training data to use
    1000 // Number of testing data to use
);

// Define weights and bias
double w[mnist.imgSize];
double b;

// Function used to compute the dot product of the weight matrix and the pixel data
double forwardPass(double *x) {
    double sum = 0;
    for (int i = 0; i < mnist.imgSize; ++i) {
        sum += w[i] * x[i];
    }
    sum += b;
    return sum;
}

// Function used to train the perceptron
void train() {
    for (auto&& digit : mnist.trainData) {
        double pred = forwardPass(digit.pixels);
        if (pred > 0 && digit.label == testNums[0]) {
            for (int i = 0; i < mnist.imgSize; ++i) {
                w[i] -= ALPHA * digit.pixels[i];
            }
            b -= ALPHA;
        }
        else if (pred < 0 && digit.label == testNums[1]) {
            for (int i = 0; i < mnist.imgSize; ++i) {
                w[i] += ALPHA * digit.pixels[i];
            }
            b += ALPHA;
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
    mnist.selectData(testNums);

    // Train model
    chrono::steady_clock::time_point t0 = chrono::steady_clock::now();
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
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    cout << "Training took " << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count() << " ms" << endl;

    // Test model
    int truePos = 0;
    
    int trueNeg = 0;
    int falsePos = 0;
    int falseNeg = 0;
    double accuracy = 0;
    for (auto&& digit : mnist.testData) {
        double pred = forwardPass(digit.pixels);

        if (digit.label == testNums[0]) {
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
    accuracy = 100.0 * (truePos + trueNeg) / mnist.testData.size();
    cout << truePos + trueNeg << "/" << mnist.testData.size() << " correct" << endl;
    cout << "The model is " << accuracy << "%" << " accurate" << endl;

    return 0;
}