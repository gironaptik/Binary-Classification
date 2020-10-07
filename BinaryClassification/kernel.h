#pragma once

#include "Perceptron.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//Calculating each point from pointsArray
__device__ double pointCalc(double * dev_weights, double * dev_parameters, int * dev_dimensionSize);

//Returning the alpha value to its absolute value
__global__ void fabsAlpha(Perceptron *dev_perceptron);

//Provides a list of calculated sign discriminant functions result
__global__ void signDiscFunctionList(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_results, int *dev_dimensionSize, int *dev_numOfPoints);

//scanning for all the unclassified points at final
__global__ void finalUnclassifiedPoints(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_results, int numOfthreads, int *dev_dimensionSize);

//Counting all the nMis points at Final
__global__ void nMisCounter(int *dev_results, int numOfthreads);

//Compute the Perceptron's q
__global__ void perceptronQCalc(Perceptron *dev_perceptron, int *dev_result, int numOfPoints, int step);

//Updating current Weight vector with new unclassifie point
__global__ void learningWeight(Perceptron *dev_perceptron, Point *dev_pointsArray, int *dev_pointIndex, int *dev_dimensionSize, int *dev_results);

//CudaFree Genetric Method
void FreeCuda(void * object);

//CudaFree of Consts Attributes Genetric Method
void FreeConstanstCuda(Point *dev_pointsArray, int *dev_numOfPoints, int *dev_dimensionSize);

//cudaMalloc Generic Method
void MallocCuda(void **dev_ptr, size_t size, char *actionTitle);

//cudaMemcpy Generic Method
void CopyCuda(void* dest, void * src, size_t size, cudaMemcpyKind kind, char *actionTitle);

//MallocCuda of Consts Attributes Method
void ConstMallocCuda(int numOfPoints, int  dimensionSize, Point *pointsArray, int **dev_numOfPoints, int **dev_dimensionSize, Point **dev_pointsArray);

//cudaFree of created attributes
cudaError_t finalize(cudaError_t cudaStatus, Perceptron *dev_perceptron, int *dev_results);

//Getting perceptron with each Alpha and tries to classified it, 
//provides its weights vector and q at the end
cudaError_t InvestigateAlpha(Point *dev_pointsArray, int *dev_numOfPoints, int *dev_dimensionSize, Perceptron *perceptron, Point *pointsArray, int numOfPoints, int dimensionSize, int limit, int QC);


