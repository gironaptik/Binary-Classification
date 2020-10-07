#pragma once
#include "Perceptron.h"

//PATH OF FILES
#define InputPATH "C:\\data1.txt"
#define OutputPATH "C:\\Output.txt"


Point* readFromFile(int* numOfPoints, int* dimensionSize, double* alphaZero, double* alphaMax, int* limit, double* QC, Point* pointArray);
void writeToFile(Perceptron perceptron, double QC, int k);
