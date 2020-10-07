//***************
//By: Giron Aptik
//ID: 307863258
//***************

#include "ReadWriteFileFunctions.h"
#include "Perceptron.h"
#include "MPI_Functions.h"
#include "kernel.h"
#include <omp.h>

int main(int argc, char*argv[])
{
	Point *pointsArray;
	/***********************************************************************
	***threadPerceptorn - Perceptron struct for each alpha in the section***
	***currentPerceptron - Provide the minimum threadPerceptron possible****
	***allPerceptron - Gather all possible currentAlpha with MPI************
	***********************************************************************/
	Perceptron *currentPerceptron, *allPerceptrons, *threadPerceptron;
	int myrank, numOfPoints, procSize, dimensionSize, limit, NMis;
	double alpha, alphaMax, QC, alphaZero;
	
	//Cuda's Constants
	Point *dev_pointsArray = NULL;
	int *dev_numOfPoints = NULL;
	int *dev_dimensionSize = NULL;

	//MPI Init:
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &procSize);

	MPI_Datatype pointMPIType, perceptronMPIType;
	createPointDataType(&pointMPIType);
	creatPerceptronDataType(&perceptronMPIType);

	pointsArray = (Point*)malloc(MAX_POINTS * sizeof(Point));
	currentPerceptron = (Perceptron*)malloc(sizeof(Perceptron));
	allPerceptrons = (Perceptron*)malloc(procSize * sizeof(Perceptron));

	if (myrank == MASTER)
	{
		//Master Proc reading from file
		readFromFile(&numOfPoints, &dimensionSize, &alphaZero, &alphaMax, &limit, &QC, pointsArray);
		if (pointsArray == NULL) {
			printf("No Points to work With...\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			fflush(stdout);
		}
	}

	//Broadcast file values to all process
	MPIBroadcastParameters(myrank, procSize, &numOfPoints, &dimensionSize, &limit, &alphaZero, &alphaMax, &alpha, &QC,  pointsArray, pointMPIType);
	
	//Malloc and Copy Const value for Cuda's Work
	ConstMallocCuda(numOfPoints, dimensionSize, pointsArray, &dev_numOfPoints, &dev_dimensionSize, &dev_pointsArray);

	//Giving the percetron & alpha high q Value
	currentPerceptron->q = QC+1;
	currentPerceptron->alpha = alphaMax;

	int threadAlpha;
	//Calculate alphas chuncks for each thread
	int alphasAmount = (int)((alphaMax - alpha) / alphaZero);
	if (fmod(alphaMax,alpha) != 0)
		alphasAmount++;

	//Using OMP to send alpha for each thread with shared currentPerceptron and private ThreadPerceptorn
	#pragma omp parallel for schedule(static) shared(currentPerceptron) private(threadPerceptron)
	for(threadAlpha = 1; threadAlpha <= alphasAmount; threadAlpha++){
		//Creating perceptron for each alpha on each thread
		threadPerceptron = (Perceptron*)malloc(sizeof(Perceptron));
		threadPerceptron->alpha = alpha * threadAlpha;
		//Check with we overhead the mac alpha of the section
		if (threadPerceptron->alpha > alphaMax)
			threadPerceptron->alpha = alphaMax;
		//init Weithgs values
		initWeights(dimensionSize, threadPerceptron->weights);
		//Checking alpha using Cuda
		InvestigateAlpha(dev_pointsArray, dev_numOfPoints, dev_dimensionSize, threadPerceptron, pointsArray, numOfPoints, dimensionSize, limit, QC);
		if (threadPerceptron->q > QC)
			break;
		else
			//Looking for the minimum alphain the section
			if (currentPerceptron->alpha > threadPerceptron->alpha) {
				currentPerceptron = threadPerceptron;
			}
	}

	//Colecting all created perceptons to allPerceptrons
	MPI_Gather(currentPerceptron, 1, perceptronMPIType, allPerceptrons, 1, perceptronMPIType, MASTER, MPI_COMM_WORLD);

	//Finds the minumium alpha and print it
	if (myrank == MASTER) {
		printFinalResult(allPerceptrons, procSize, dimensionSize, QC);
	}

	//Free programs attributes (Including Cuda Consts)
	FreeConstanstCuda(dev_pointsArray,dev_numOfPoints, dev_dimensionSize);
	free(pointsArray);
	free(currentPerceptron);
	free(allPerceptrons);
	MPI_Finalize();
}
