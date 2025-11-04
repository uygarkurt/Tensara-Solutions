#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void l1_normalization(const float* X, float* Y, const size_t B, const size_t D){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= B) return;
    float sum = 0.0f;

    for (int j = 0; j < D; j++){
        sum += abs(X[i * D + j]);
    }

    for (int j = 0; j < D; j ++){
        Y[i * D + j] = X[i * D + j] / sum;
    }
}

// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t D) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;

    l1_normalization<<<blocksPerGrid, threadsPerBlock>>>(X, Y, B, D);
    cudaDeviceSynchronize();
}