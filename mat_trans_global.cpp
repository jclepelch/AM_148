#include <iostream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <omp.h>

/* Use Matrix Class! */
#include "mat.h"
#include "submat.h"
#define BLOCK_SIZE 32

__global__ void MatTransGlobalKernel( Matrix A, Matrix transA)
{	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	for (int e = 0; e < A.width; e++)
	{
		float transAValue = 0.0f;
		transAValue = A.elements[row * A.width + e];
		transA.elements[e * transA.width + col] = transAValue;
	}
}

void MatTransGlobal(const Matrix A, Matrix transA)
{
	int Gpu = 1, toDev = 1, fromDev = 2;
	
	Matrix d_A(A.width, A.height, 0, Gpu);
	d_A.load(A, toDev);

	Matrix d_transA(transA.width, transA.height, 0, Gpu);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(A.width / dimBlock.x, A.height / dimBlock.y)

	hipEvent_t start, stop;
	float elapsed_secs;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	hipEventRecord(start, 0);

	MatTransGlobalKernel << <dimGrid, dimBlock >> > (d_A, d_transA);
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&elapsed_secs, start, stop);
	std::cout << " Naive GPU MatMul Time = " << elapsed_secs << "ms" << std::endl;
	// Read C from device memory 
	transA.load(d_transA, fromDev);
	// Free device memory 
	d_A.dealloc(Gpu);
	d_transA.dealloc(Gpu);
}