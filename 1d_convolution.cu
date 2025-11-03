#include <cuda_runtime.h>

__global__ void convolution1d(const float* A, const float* B, float* C, size_t N, size_t K, int shift){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float sum = 0.0f;

    for (int j = 0; j < K; j++){
        int input_idx = i + j - shift;
        if (input_idx >= 0 && input_idx < N){
            sum += A[input_idx] * B[j];
        }
    }
    C[i] = sum;
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K){
    int shift = (K - 1)/2;
    // N += shift * 2;

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    convolution1d<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K, shift);
    cudaDeviceSynchronize();
}
