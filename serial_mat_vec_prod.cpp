#include <iostream>
#include <chrono>


void serialMatVecProd(const float A[][], const float x[], b[], const N)
{
	for (int i = 0; i < N; i++)
	{
		float bvalue = 0.0f;
		for (int j = 0; j < N; j++)
		{
			b[j] += A[i][j] * x[j];
		}
	}
}

void runtime(int N)
{

	float *A, * x;
	A = new float[N][N];
	x = new float[N];
	using std::chrono::high_resolution_clock;
	using std::chrono::duration_cast;
	using std::chrono::duration;
	using std::chrono::milliseconds;

	auto t1 = high_resolution_clock::now();

	serialMatVecProd(A, x, b, N);

	auto t2 = high_resolution_clock::now();
	
	duration<double, std::milli> ms_double = t2 - t1;

	std::cout << "The runtime for the serial matrix-vector product of a matrix and vector of size " << N << " was " << ms_double.count() << "ms\n";

}

int main()
{
	runtime(16);
	runtime(128);
	runtime(1024);
	runtime(2048);
	runtime(65536);
	return 0;
}