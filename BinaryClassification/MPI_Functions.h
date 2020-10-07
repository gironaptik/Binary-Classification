#pragma once

#include "Perceptron.h"
#include "ReadWriteFileFunctions.h"
#include <mpi.h>

#define MASTER 0
#define DATA_TAG 0
#define PERCEPTRON_ATTRIBUTES_COUNT 3		//Weigths, q, alpha
#define POINT_ATTRIBUTES_COUNT 2			//parameters, label

//Init Perceptron & Point MPITypes
void creatPerceptronDataType(MPI_Datatype* perceptronMPIType);
void createPointDataType(MPI_Datatype* pointMPIType);
void printFinalResult(Perceptron *allPerceptrons, int numProcs, int dimensionSize, double QC);

//Broadcast custom function- broadcast all file data to all processes
void MPIBroadcastParameters(
	int rank, int numprocs,
	int *numOfPoints, int *dimensionSize, int *limit, double *alphaZero, double *alphaMax, double * alpha, double *QC,
	Point* points, MPI_Datatype  pointMPIType);

//Finding the minimum alpha from allPerceptron
int MinAlpha(Perceptron* allPerceptrons, double QC, int alphasPerProc);
