#include "Perceptron.h"
#include <omp.h>

//Initalize Weights Vector to Zero
void initWeights(int dimensionSize, double* weights) {	
	#pragma omp parallel for 
	for (int i = 0; i < dimensionSize; i++)
		weights[i] = 0;
	//Bias Value
	weights[dimensionSize] = 0;	
}

//Check if alpha exist & Prints final results
void printResults(Perceptron perceptron, double QC, int dimensionSize) {
	if (perceptron.q < QC) {
		printf("Alpha minimum = %lf  q = %lf\n", perceptron.alpha, perceptron.q);
		for (int i = 0; i <= dimensionSize; i++)
			printf("w%d) %lf\n", i + 1, perceptron.weights[i]);
	}
	else {
		printf("Sorry, Alpha doesn't exist\n");
	}
}