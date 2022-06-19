#pragma once

#include <vector>
class mnistreader {
    private:
        struct mnistDigit {
            int label;
            double pixels[784];
        };

        std::vector<mnistDigit> fullTrainData;
        std::vector<mnistDigit> fullTestData;

        int trainSize;
        int testSize;
    
    public:
        std::vector<mnistDigit> trainData;
        std::vector<mnistDigit> testData;

        static const int imgSize = 784;

        mnistreader(
            const char* trainImgLoc,
            const char* trainLblLoc,
            const char* testImgLoc,
            const char* testLblLoc,
            int nTrain,
            int nTest
        );

        void selectData(std::vector<int> list);
};