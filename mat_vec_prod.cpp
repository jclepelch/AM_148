#include <iostream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <omp.h>
#include <vector>

/* Use Matrix Class! */
#include "mat.h"

#define BLOCK_SIZE 32;

using namespace std;

class subMatrix
{
public:
	int width;
	int stride;
	float* elements;
	float* elementsVector
	__device__ subMatrix(Matrix A, int sub_size, int row, int col)
	{
		width = sub_size;
		stride = A.stride;
		elements = &A.elements[stride * width * row + width * col];
	}

	__device__ inline void SetElem(const int row, const int col, const float value)
	{
		elements[row * stride + col] = value;
	}

	__device__ inline float GetElem(const int row, const int col)
	{
		return elements[row * stride + col];
	}

};

__global__ void MatVecProdKernel(const Matrix A, const Matrix x, Matrix b)
{
	__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float xs[BLOCK_SIZE];


	int blockRow = blockIdx.y;
	int blockCol = blockIdx.x;

	//Thread block computes 32 entries of b all in one column
	subMatrix bSub(b, BLOCK_SIZE, blockRow, 0);
	
	float bVal = 0.f;

	int row = threadIdx.y;
	int col = threadIdx.x;

	for (int m = 0; m < (A.width / BLOCK_SIZE); m++)
	{
		subMatrix ASub(A, BLOCK_SIZE, blockRow, m);

		subMatrix xSub(x, BLOCK_SIZE, m, 0);

		As[row][col] = ASub.GetElem(row, col); 
		xs[row][col] = xSub.GetElem(row, 0);

		__syncthreads();

		for (int e = 0; e < BLOCK_SIZE; e++)
		{
			bVal += As[row][e] * xs[e];
		}

		__syncthreads();

		bSub.SetElem(row, 0, bVal);
	}
}

void MatVecProd(const Matrix A, const Matrix x, Matrix b)
{
	int Gpu = 1;
	int toDev = 1, fromDev = 2;
	//Load A and B to device memory 
	//Allocate vector b
	Matrix d_A(A.width, A.height, A.stride, Gpu);
	Matrix d_x(x.width, x.height, x.stride, Gpu);
	Matrix d_b(b.width, b.height, b.stride, Gpu);
	d_A.load(A, toDev);
	d_x.load(x, toDev);

	// Invoke Kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
	//Use HIP Events for timing
	hipEvent_t start, stop;
	float time;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	hipEventRecord(start, 0);

	MatMulKernel << <dimGrid, dimBlock >> > (d_A, d_x, d_b);
	hipEventRecord(stop, 0);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&time, start, stop);
	std::cout << " Shared Memory Matrix-Vector Product time =" << '\t' << time << "ms" << std::endl;

	// Read b from Device memory 
	b.load(d_b, fromDev);

	//Free device memory 
	d_A.dealloc(Gpu);
	d_x.dealloc(Gpu);
	d_b.dealloc(Gpu);
}




int run(int N)
{
	
}

int main()
{
	run(16);
	run(128);
	run(1024);
	run(2048);
	run(65536);
}


