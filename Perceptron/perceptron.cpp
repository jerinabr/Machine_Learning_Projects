#include <iostream>
#include <fstream>

using namespace std;

const int IMG_SIZE = 28 * 28;
const int NUM_IMAGES = 1000;
const double ALPHA = 0.01;

double w[IMG_SIZE + 1];

double trainIMG[NUM_IMAGES][IMG_SIZE + 1];
int trainLBL[NUM_IMAGES];

int totSamples;

// Function used to compute the dot product of the weight matrix and the pixel data
double dotProduct(double *x) {
    double sum;
    for (int i = 0; i < IMG_SIZE + 1; ++i) {
        sum += w[i] * x[i];
    }
    return sum;
}

void readMNIST() {
    double fullTrainIMG[NUM_IMAGES][IMG_SIZE + 1];
    unsigned char fullTrainLBL[NUM_IMAGES];
    
    // Read training images
    ifstream trainImgFile("train-images-idx3-ubyte", ifstream::binary);
    trainImgFile.seekg(16); // Skip metadata
    for (int i = 0; i < NUM_IMAGES * IMG_SIZE; ++i) {
        unsigned char *temp = new unsigned char;
        trainImgFile.read((char*) temp, 1);
        fullTrainIMG[i / IMG_SIZE][i % IMG_SIZE] = *temp / 255.0;
    }
    trainImgFile.close();

    // Read training labels
    ifstream trainLblFile("train-labels-idx1-ubyte", ifstream::binary);
    trainLblFile.seekg(8); // Skip metadata
    for (int i = 0; i < NUM_IMAGES; ++i) {
        trainLblFile.read((char*) (fullTrainLBL + i), 1);
    }
    trainLblFile.close();

    // Select 2 and 6
    int j = 0;
    for (int i = 0; i < NUM_IMAGES; ++i) {
        if (fullTrainLBL[i] == 2 || fullTrainLBL[i] == 6) {
            copy(begin(fullTrainIMG[i]), end(fullTrainIMG[i]), begin(trainIMG[j]));
            trainIMG[j][IMG_SIZE] = 1;
            trainLBL[j] = (fullTrainLBL[i] == 2 ? -1 : 1);
            ++j;
        }
    }
    totSamples = j;
}

void train() {
    for (int i = 0; i < totSamples; ++i) {
        double pred = dotProduct(trainIMG[i]) * trainLBL[i];
        if (pred <= 0) {
            for (int j = 0; j < IMG_SIZE + 1; ++j) {
                w[j] += ALPHA * trainLBL[i] * trainIMG[i][j];
            }
        }
    }
}

int main() {
    // Initialize weights and bias
    for (int i = 0; i < IMG_SIZE + 1; ++i) {
        w[i] = (double) rand() / RAND_MAX;
    }

    // Read MNIST dataset
    readMNIST();

    // Train model
    ofstream wData;
    wData.open("wData.csv");
    for (int i = 0; i < 1000; ++i) {
        train();
        for (int j = 0; j < IMG_SIZE + 1; ++j) {
            wData << w[j] << ",";
        }
        wData << "\n";
    }
    wData.close();

    return 0;
}