#include "ReadWriteFileFunctions.h"
#include "Perceptron.h"


Point* readFromFile(int* numOfPoints, int* dimensionSize, double* alphaZero, double* alphaMax, int* limit, double* QC, Point* pointArray) {
	FILE* f = fopen(InputPATH, "r");
	*numOfPoints = 0;
	int i, j = 0;
	// Checking possible file opening failure
	if (f == NULL) {
		printf("Failed opening the file. Exiting!\n");
		exit(1);
	}
	else {
		fscanf(f, "%d", numOfPoints);
		fscanf(f, "%d", dimensionSize);
		fscanf(f, "%lf", alphaZero);
		fscanf(f, "%lf", alphaMax);
		fscanf(f, "%d", limit);
		fscanf(f, "%lf", QC);

		printf("n: %d , dim:%d, a:%f, amax:%f, limit:%d, QC:%f \n", *numOfPoints,
			*dimensionSize, *alphaZero, *alphaMax, *limit, *QC);

		//read all points deatils:
		for (int i = 0; i < *numOfPoints; i++) {
			for (int j = 0; j < *dimensionSize; j++)
				fscanf(f, "%lf", &(pointArray[i].parameters[j]));
			fscanf(f, "%d", &(pointArray[i].label));
		}
		fclose(f);
	}
	return pointArray;
}

void writeToFile(Perceptron perceptron, double QC, int dimensionSize) {
	FILE *outPutFile = fopen(OutputPATH, "w");
	if (outPutFile == NULL)
	{
		printf("Can't open the file");
		exit(1);
	}
	if (perceptron.q < QC)
	{
		fprintf(outPutFile, "Alpha minimum  = %lf q = %lf\n", perceptron.alpha, perceptron.q);
		for (int i = 0; i <= dimensionSize; i++)
			fprintf(outPutFile, "w%d) %.9f\n", i + 1, perceptron.weights[i]);
	}
	else
		fprintf(outPutFile, "Sorry, Alpha doesn't exist");
	fclose(outPutFile);
}
