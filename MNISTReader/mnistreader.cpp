#include <iostream>
#include <fstream>
#include <algorithm>
#include "mnistreader.hpp"

#define NUM_TRAINING 60000
#define NUM_TESTING 10000

mnistreader::mnistreader(const char* trainImgLoc, const char* trainLblLoc, const char* testImgLoc, const char* testLblLoc, int nTrain, int nTest) {
    // Initialize training and testing data vectors
    trainSize = (nTrain ? nTrain : NUM_TRAINING);
    testSize = (nTest ? nTest : NUM_TESTING);

    fullTrainData.reserve(NUM_TRAINING);
    fullTestData.reserve(NUM_TESTING);

    trainData.reserve(trainSize);
    testData.reserve(testSize);

    // Read training data from image and label files
    std::ifstream trainImgFile(trainImgLoc, std::ios::binary);
    std::ifstream trainLblFile(trainLblLoc, std::ios::binary);
    if (trainImgFile.is_open() && trainLblFile.is_open()) {
        // Skip file metadata
        trainImgFile.seekg(16);
        trainLblFile.seekg(8);

        unsigned char* tempLbl = new unsigned char[1];
        unsigned char* tempImg = new unsigned char[imgSize];
        for (int i = 0; i < NUM_TRAINING; ++i) {
            mnistDigit digit;

            // Read file data into placeholders
            trainLblFile.read((char*) tempLbl, 1);
            trainImgFile.read((char*) tempImg, imgSize);

            // Use placeholder values for digit struct
            digit.label = (int) *tempLbl;
            for (int j = 0; j < imgSize; ++j) {
                digit.pixels[j] = (double) tempImg[j] / 255.0; // Normalize pixel values
            }

            fullTrainData.push_back(digit);
        }
    }
    trainImgFile.close();
    trainLblFile.close();

    // Read testing data from image and label files
    std::ifstream testImgFile(testImgLoc, std::ios::binary);
    std::ifstream testLblFile(testLblLoc, std::ios::binary);
    if (testImgFile.is_open() && testLblFile.is_open()) {
        // Skip file metadata
        testImgFile.seekg(16);
        testLblFile.seekg(8);

        unsigned char* tempLbl = new unsigned char[1];
        unsigned char* tempImg = new unsigned char[imgSize];
        for (int i = 0; i < NUM_TESTING; ++i) {
            mnistDigit digit;

            // Read file data into placeholders
            testLblFile.read((char*) tempLbl, 1);
            testImgFile.read((char*) tempImg, imgSize);

            // Use placeholder values for digit struct
            digit.label = (int) *tempLbl;
            for (int j = 0; j < imgSize; ++j) {
                digit.pixels[j] = (double) tempImg[j] / 255.0; // Normalize pixel values
            }

            fullTestData.push_back(digit);
        }
    }
    testImgFile.close();
    testLblFile.close();
}

void mnistreader::selectData(std::vector<int> list) {
    // Modify training data
    int numSelected = 0;
    for (auto&& digit : fullTrainData) {
        // Check if the digit is in the list of digits
        if (std::find(list.begin(), list.end(), digit.label) != list.end()) {
            // Add digit to the training whitelist
            trainData.push_back(digit);
            if (++numSelected == trainSize) break;
        }
    }

    // Modify testing data
    numSelected = 0;
    for (auto&& digit : fullTestData) {
        // Check if the digit is in the list of digits
        if (std::find(list.begin(), list.end(), digit.label) != list.end()) {
            // Add digit to the testing whitelist
            testData.push_back(digit);
            if (++numSelected == testSize) break;
        }
    }
}