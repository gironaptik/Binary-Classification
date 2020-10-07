#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#define MAX_COORDINATES 21	//	Highet dimension size +1 for constant (bias / w0)
#define MAX_POINTS 500000	//	Highest points amount possible
#define MAX_ALPHAS 100		//	Highest alpha value possible
#define SIGN(x) (x >= 0 ? 1 : -1) //Checking Functino Size


typedef struct 
{
	double parameters[MAX_COORDINATES];
	int label;
} Point;

typedef struct
{
	double alpha;
	double weights[MAX_COORDINATES];
	double q;
} Perceptron;

void initWeights(int dimension, double* weights);

//Prints results to console and the Outputfile
void printResults(Perceptron perceptron, double QC, int k);
