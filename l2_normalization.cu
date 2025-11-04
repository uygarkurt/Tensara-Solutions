#include <cuda_runtime.h>
#include <math.h>

__global__ void l2_normalization(const float* X, float* Y, const size_t B, const size_t D)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= B) return;
    float sum = 0.0f;

    for (int j = 0; j < D; j++){
        sum += pow(X[i * D + j], 2);
    }
    float res = sqrt(sum);

    for (int j = 0; j < D; j++){
        Y[i * D + j] = X[i * D + j] / res;
    }
}

// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    l2_normalization<<<blocksPerGrid, threadsPerBlock>>>(X, Y, B, D);
    cudaDeviceSynchronize();
}