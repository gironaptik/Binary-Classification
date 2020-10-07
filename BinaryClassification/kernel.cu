#include "kernel.h"
#include <stdio.h>

#define MAX_NUM_OF_THREADS 1000

__device__ double pointCalc(double * dev_weights, double * dev_parameters, int * dev_dimensionSize)
{
	int i;
	double sum = 0;
	for (i = 0; i < *dev_dimensionSize; i++)
		sum += dev_weights[i] * dev_parameters[i];
	//Bias Calc
	sum += dev_weights[i];	
	return sum;
}

__global__ void fabsAlpha(Perceptron *dev_perceptron)
{
	dev_perceptron->alpha = fabs(dev_perceptron->alpha);
}

__global__ void signDiscFunctionList(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_results, int *dev_dimensionSize, int *dev_numOfPoints)
{
	int i = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	if (i < *dev_numOfPoints)
	{
		// calaculate fx 
		double fx = pointCalc(dev_perceptron->weights, dev_pointsArray[i].parameters, dev_dimensionSize);
		int sign = SIGN(fx);
		if (dev_pointsArray[i].label != sign) { 
			sign = (dev_pointsArray[i].label - sign) / 2;
			dev_results[i] = sign;
		}
		else
			dev_results[i] = 0;
	}

}

__global__ void finalUnclassifiedPoints(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_results, int numOfthreads, int *dev_dimensionSize)
{
	int thread_index = threadIdx.x;
	int block_index = blockIdx.x;
	int index = thread_index + block_index * numOfthreads;
	int error = SIGN(pointCalc(dev_perceptron->weights, dev_pointsArray[index].parameters, dev_dimensionSize)) - dev_pointsArray[index].label;
	dev_results[index] = error;
}

__global__ void nMisCounter(int *dev_results, int numOfthreads)
{
	int index = threadIdx.x *numOfthreads, i;
	dev_results[index] = dev_results[index] != 0 ? 1 : 0;
	for (i = index + 1; i < index + numOfthreads; i++)
		if (dev_results[i] != 0)
			dev_results[index]++;
}

__global__ void perceptronQCalc(Perceptron *dev_perceptron, int *dev_result, int numOfPoints, int step) {
	for (int i = step; i < numOfPoints; i += step)
		dev_result[0] += dev_result[i];
	dev_perceptron->q = (double)dev_result[0] / (double)numOfPoints;
}

__global__ void learningWeight(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_pointIndex, int *dev_dimensionSize, int *dev_results)
{
	//Alpha * SignFunc
	dev_perceptron->alpha = dev_perceptron->alpha*dev_results[*dev_pointIndex];

	int i = threadIdx.x;
	if (i == *dev_dimensionSize) // Bias
		dev_perceptron->weights[i] += dev_perceptron->alpha;
	else
		dev_perceptron->weights[i] += dev_perceptron->alpha*dev_pointsArray[*dev_pointIndex].parameters[i];
}


//CudaFree Genetric Method
void FreeCuda(void * object)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(object);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaFree failed!\n");
}

//CudaFree of Consts Attributes Genetric Method
void FreeConstanstCuda(Point *dev_pointsArray, int *dev_numOfPoints, int *dev_dimensionSize)
{
	FreeCuda(dev_pointsArray);
	FreeCuda(dev_numOfPoints);
	FreeCuda(dev_dimensionSize);
}

//cudaMalloc Generic Method
void MallocCuda(void **dev_ptr, size_t size, char *actionTitle)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc(dev_ptr, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! error_label : %s\n", actionTitle);
		FreeCuda(*dev_ptr);
	}
}

//cudaMemcpy Generic Method
void CopyCuda(void* dest, void * src, size_t size, cudaMemcpyKind kind, char *actionTitle)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dest, src, size, kind);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed! error_label : %s\n", actionTitle);
}

//MallocCuda of Consts Attributes Method
void ConstMallocCuda(int numOfPoints, int  dimensionSize, Point *pointsArray, int **dev_numOfPoints, int **dev_dimensionSize, Point **dev_pointsArray)
{

	MallocCuda((void**)&(*dev_pointsArray), sizeof(Point)* numOfPoints, "Malloc pointsArray");
	CopyCuda(*dev_pointsArray, pointsArray, sizeof(Point)*numOfPoints, cudaMemcpyHostToDevice, "Copy pointsArray");
	MallocCuda((void**)&(*dev_numOfPoints), sizeof(int), "Malloc numOfPoints Size");
	MallocCuda((void**)&(*dev_dimensionSize), sizeof(int), "Malloc Dimension Size");
	CopyCuda((*dev_numOfPoints), &numOfPoints, 1, cudaMemcpyHostToDevice, "Copy numOfPoints");
	CopyCuda((*dev_dimensionSize), &dimensionSize, 1, cudaMemcpyHostToDevice, "Copy dimension Size");
}

//cudaFree of created attributes
cudaError_t finalize(cudaError_t cudaStatus, Perceptron *dev_perceptron, int *dev_results)
{
	cudaFree(dev_perceptron);
	cudaFree(dev_results);
	return cudaStatus;
}

//Getting perceptron with each Alpha and tries to classified it, 
//provides its weights vector and q at the end
cudaError_t InvestigateAlpha(Point *dev_pointsArray, int *dev_numOfPoints, int *dev_dimensionSize, Perceptron *perceptron,	Point *pointsArray, int numOfPoints, int dimensionSize, int limit, int QC)
{

	int *results = (int*)malloc((numOfPoints) * sizeof(int*));
	Perceptron *dev_perceptron = NULL;
	int *dev_results = NULL;
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return finalize(cudaStatus, dev_perceptron, dev_results);
	}

	//Check if we can move it to the main
	MallocCuda((void**)&dev_perceptron, sizeof(Perceptron), "Perceptron Malloc");
	MallocCuda((void**)&dev_results, sizeof(int)*numOfPoints, "Mislead Malloc");
	cudaMemset(dev_results, 0, sizeof(int)* (numOfPoints));
	CopyCuda(dev_perceptron, perceptron, sizeof(Perceptron), cudaMemcpyHostToDevice, "Copy Perceptron");
	
	int iterationLimit = 0;

	int blocks = numOfPoints / MAX_NUM_OF_THREADS > 0 ? numOfPoints / MAX_NUM_OF_THREADS : 1;
	int threads = numOfPoints / MAX_NUM_OF_THREADS > 0 ? MAX_NUM_OF_THREADS : numOfPoints;
	
	while (iterationLimit < limit)
	{
		//Check if we need to add +1 to the blocks
		signDiscFunctionList << < blocks, threads >> > (dev_perceptron, dev_pointsArray, dev_results, dev_dimensionSize, dev_numOfPoints);
		
		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "signDiscFunctionList launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateResultsKernel!\n", cudaStatus);
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}

		CopyCuda(results, dev_results, (numOfPoints) * sizeof(int), cudaMemcpyDeviceToHost, "Copy MisLeads");
		int unclassifiedPoint;

		//Checking for the first point that unclassified
		for (unclassifiedPoint = 0; unclassifiedPoint < numOfPoints; unclassifiedPoint++)
		{
			if (results[unclassifiedPoint] == -1 || results[unclassifiedPoint] == 1)
				break;
		}
		if (unclassifiedPoint == numOfPoints)
			break;
		else
		{
			int *dev_unclassifiedPoint = NULL;
			MallocCuda((void**)&dev_unclassifiedPoint, sizeof(int), "dev_unclassifiedPoint Malloc");
			CopyCuda(dev_unclassifiedPoint, &unclassifiedPoint, sizeof(int), cudaMemcpyHostToDevice, "Copy dev_unclassifiedPoint");

			learningWeight << <1, dimensionSize+1 >> > (dev_perceptron, dev_pointsArray, dev_unclassifiedPoint, dev_dimensionSize, dev_results);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "learningWeight launch failed: %s\n", cudaGetErrorString(cudaStatus));
				return finalize(cudaStatus, dev_perceptron, dev_results);
			}

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching TrainWeight!\n", cudaStatus);
				return finalize(cudaStatus, dev_perceptron, dev_results);
			}

			//Returning alpha to its original value
			fabsAlpha << <1, 1 >> > (dev_perceptron);

			FreeCuda(dev_unclassifiedPoint);
		}
		iterationLimit++;
	}

		finalUnclassifiedPoints << <blocks, threads >> > (dev_perceptron, dev_pointsArray, dev_results, threads, dev_dimensionSize);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "finalUnclassifiedPoints launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}
	
		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateResultsKernel!\n", cudaStatus);
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}

		nMisCounter << < 1, blocks >> > (dev_results, threads);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "nMisCounter  launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}
	
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching sumOfMisses !\n", cudaStatus);
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}
	
		perceptronQCalc << < 1, 1 >> > (dev_perceptron, dev_results, numOfPoints, threads);
	
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "perceptronQCalc  launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}
	
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching computeQuality !\n", cudaStatus);
			return finalize(cudaStatus, dev_perceptron, dev_results);
		}

	//Copy the perceptron to the Host
	CopyCuda(perceptron, dev_perceptron, sizeof(Perceptron), cudaMemcpyDeviceToHost, "Copy dev_perceptron to perceptron");

	//Free Memory
	finalize(cudaStatus, dev_perceptron, dev_results);
	free(results);
}

