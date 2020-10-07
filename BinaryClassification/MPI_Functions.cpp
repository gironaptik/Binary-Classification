#include "MPI_Functions.h"

void createPointDataType(MPI_Datatype* pointMPIType) {
	Point point;
	// Point Attributes properties
	MPI_Datatype type[POINT_ATTRIBUTES_COUNT]{ MPI_DOUBLE, MPI_INT}; 
	int blocklen[POINT_ATTRIBUTES_COUNT] = { MAX_COORDINATES , 1};
	MPI_Aint disp[POINT_ATTRIBUTES_COUNT];

	//	Memory calculation offsets
	disp[0] = (char *)&point.parameters - (char *)&point;
	disp[1] = (char *)&point.label - (char *)&point;

	MPI_Type_create_struct(POINT_ATTRIBUTES_COUNT, blocklen, disp, type, pointMPIType);
	MPI_Type_commit(pointMPIType);
}

void creatPerceptronDataType(MPI_Datatype* perceptronMPIType) {

	Perceptron perceptron;
	MPI_Datatype type[PERCEPTRON_ATTRIBUTES_COUNT] = { MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE };
	int blocklen[PERCEPTRON_ATTRIBUTES_COUNT] = { 1, MAX_COORDINATES, 1 };
	MPI_Aint disp[PERCEPTRON_ATTRIBUTES_COUNT];

	//	Memory calculation offsets
	disp[0] = (char *)&perceptron.alpha - (char *)&perceptron;
	disp[1] = (char *)&perceptron.weights - (char *)&perceptron;
	disp[2] = (char *)&perceptron.q - (char *)&perceptron;

	MPI_Type_create_struct(PERCEPTRON_ATTRIBUTES_COUNT, blocklen, disp, type, perceptronMPIType);
	MPI_Type_commit(perceptronMPIType);
	}

	void MPIBroadcastParameters(int rank, int numprocs,int *n, int *k, int *limit, 
		double *alphaZero, double *alphaMax, double * alpha, double *QC, Point* points, MPI_Datatype  pointMPIType) 
		{
		MPI_Status status;
		MPI_Bcast(limit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(alphaZero, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(k, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(QC, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(points, *n, pointMPIType, MASTER, MPI_COMM_WORLD);

		//Calculate the alphas for each proc
		if (rank == MASTER) {	
			int alphaAmount = (int)(*alphaMax / *alphaZero);
			int alphaPerProc = (alphaAmount / numprocs);	//Calculate how many alpha to send per proc
			int checkAlpha = (alphaAmount % numprocs);		//Checking for reminder
			double alphaStep = (double)alphaPerProc*(*alphaZero) - *alphaZero;
			double alphaStart, alphaFinal = *alphaZero + alphaStep;
			alphaStart = *alpha = *alphaZero;
			for (int i = 1; i < numprocs; i++)		//Slave proc only
			{
				alphaStart = alphaFinal + *alphaZero;
				if (i >= numprocs - checkAlpha)
					alphaFinal = alphaStart + alphaStep + *alphaZero;
				else
					alphaFinal = alphaStart + alphaStep;
				if (i == numprocs - 1)
					alphaFinal = *alphaMax;

				//Sending to each process it's start and end alpha
				MPI_Send(&alphaStart, 1, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD);
				MPI_Send(&alphaFinal, 1, MPI_DOUBLE, i, DATA_TAG, MPI_COMM_WORLD);
			}
			*alphaMax = *alphaZero + alphaStep;
		}
		else
		{
			//Recieving alpha's section from the master
			MPI_Recv(alpha, 1, MPI_DOUBLE, MASTER, DATA_TAG, MPI_COMM_WORLD, &status);
			MPI_Recv(alphaMax, 1, MPI_DOUBLE, MASTER, DATA_TAG, MPI_COMM_WORLD, &status);
		}
	}

	//	Searching the minimal Alpha from all of the Alpha's that found
	int MinAlpha(Perceptron* allPerceptrons, double QC, int procSize) { 
		int minIndex = procSize - 1;
		#pragma omp parallel for
		for (int i = 0; i < procSize; i++) {
			if (allPerceptrons[i].q < QC && i < minIndex)
				minIndex = i;
		}
		return minIndex;
	}

	//Printing the final results - MASTER proc only
	void printFinalResult(Perceptron *allPerceptrons, int procSize, int dimensionSize, double QC) {
		int minIndex;
		minIndex = MinAlpha(allPerceptrons, QC, procSize);
		printResults(allPerceptrons[minIndex], QC, dimensionSize);
		writeToFile(allPerceptrons[minIndex], QC, dimensionSize);
	}

	
