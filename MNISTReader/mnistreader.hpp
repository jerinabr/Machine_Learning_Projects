#pragma once

class mnistreader {
    private:
        double** fullTrainImgData;
        int* fullTrainLblData;

        double** fullTestImgData;
        int* fullTestLblData;

        int fullTrainSize;
        int fullTestSize;

        bool numInList(int n, int* list, int size);
        void cloneImg(double* target, double* source);
    
    public:
        double** trainImgData;
        int* trainLblData;
        int trainSize;

        double** testImgData;
        int* testLblData;
        int testSize;

        static const int imgSize = 784;

        mnistreader(
            const char* trainImgLoc,
            const char* trainLblLoc,
            const char* testImgLoc,
            const char* testLblLoc,
            int nTrain,
            int nTest,
            bool normalize
        );

        void selectData(int* list, int size);
};