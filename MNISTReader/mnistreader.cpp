#include <iostream>
#include <fstream>
#include "mnistreader.hpp"

using namespace std;

#define NUM_TRAINING 60000
#define NUM_TEST 10000

bool mnistreader::numInList(int n, int* list, int size) {
    for (int i = 0; i < size; ++i) {
        if (n == list[i]) {
            return true;
        }
    }
    return false;
}

void mnistreader::cloneImg(double* target, double* source) {
    for (int i = 0; i < imgSize; ++i) {
        target[i] = source[i];
    }
}

mnistreader::mnistreader(const char* trainImgLoc, const char* trainLblLoc, const char* testImgLoc, const char* testLblLoc, int nTrain, int nTest, bool normalize) {
    // Initialize training data arrays
    fullTrainSize = (nTrain == 0 ? NUM_TRAINING : nTrain);
    trainSize = fullTrainSize;

    fullTrainImgData = (double**) malloc(fullTrainSize * sizeof(double*));
    trainImgData = (double**) malloc(fullTrainSize * sizeof(double*));
    for (int i = 0; i < fullTrainSize; ++i) {
        fullTrainImgData[i] = (double*) malloc(imgSize * sizeof(double));
        trainImgData[i] = (double*) malloc(imgSize * sizeof(double));
    }

    fullTrainLblData = (int*) malloc(fullTrainSize * sizeof(int));
    trainLblData = (int*) malloc(fullTrainSize * sizeof(int));

    // Initialize test data arrays
    fullTestSize = (nTest == 0 ? NUM_TEST : nTest);
    testSize = fullTestSize;

    fullTestImgData = (double**) malloc(fullTestSize * sizeof(double*));
    testImgData = (double**) malloc(fullTestSize * sizeof(double*));
    for (int i = 0; i < fullTestSize; ++i) {
        fullTestImgData[i] = (double*) malloc(imgSize * sizeof(double));
        testImgData[i] = (double*) malloc(imgSize * sizeof(double));
    }

    fullTestLblData = (int*) malloc(fullTestSize * sizeof(int));
    testLblData = (int*) malloc(fullTestSize * sizeof(int));

    // Initialize temporary data vars
    unsigned char* temp = (unsigned char*) malloc(1);
    double val;

    // Read training images
    ifstream trainImgFile(trainImgLoc, ifstream::binary);
    trainImgFile.seekg(16); // Skip metadata

    for (int i = 0; i < fullTrainSize * imgSize; ++i) {
        trainImgFile.read((char*) temp, 1);

        val = (normalize ? *temp / 255.0 : (double) *temp);

        fullTrainImgData[i / imgSize][i % imgSize] = val;
        trainImgData[i / imgSize][i % imgSize] = val;
    }

    trainImgFile.close();

    // Read training labels
    ifstream trainLblFile(trainLblLoc, ifstream::binary);
    trainLblFile.seekg(8); // Skip metadata

    for (int i = 0; i < fullTrainSize; ++i) {
        trainLblFile.read((char*) temp, 1);

        fullTrainLblData[i] = (int) *temp;
        trainLblData[i] = (int) *temp;
    }

    trainLblFile.close();

    // Read test images
    ifstream testImgFile(testImgLoc, ifstream::binary);
    testImgFile.seekg(16); // Skip metadata

    for (int i = 0; i < fullTestSize * imgSize; ++i) {
        testImgFile.read((char*) temp, 1);


        val = (normalize ? *temp / 255.0 : (double) *temp);

        fullTestImgData[i / imgSize][i % imgSize] = val;
        testImgData[i / imgSize][i % imgSize] = val;
    }

    testImgFile.close();

    // Read test labels
    ifstream testLblFile(testLblLoc, ifstream::binary);
    testLblFile.seekg(8); // Skip metadata

    for (int i = 0; i < fullTestSize; ++i) {
        testLblFile.read((char*) temp, 1);

        fullTestLblData[i] = (int) *temp;
        testLblData[i] = (int) *temp;
    }

    testLblFile.close();

    free(temp);
}

void mnistreader::selectData(int* list, int size) {
    // Modify training data
    {
        int j = 0;
        for (int i = 0; i < fullTrainSize; ++i) {
            if (numInList(fullTrainLblData[i], list, size)) {
                cloneImg(trainImgData[j], fullTrainImgData[i]);
                trainLblData[j] = fullTrainLblData[i];
                ++j;
            }
        }
        trainSize = j;
    }

    // Modify testing data
    {
        int j = 0;
        for (int i = 0; i < fullTestSize; ++i) {
            if (numInList(fullTestLblData[i], list, size)) {
                cloneImg(testImgData[j], fullTestImgData[i]);
                testLblData[j] = fullTestLblData[i];
                ++j;
            }
        }
        testSize = j;
    }
}