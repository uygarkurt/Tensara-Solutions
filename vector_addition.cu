// YouTube Tutorial: https://youtu.be/yzq5zlR3Y1c

#include <cuda_runtime.h>

__global__ void vector_add(const float* d_input1, const float* d_input2, float* d_output, size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n){
        d_output[i] = d_input1[i] + d_input2[i];
    }
}

// Note: d_input1, d_input2, d_output are all device pointers to float32 arrays
extern "C" void solution(const float* d_input1, const float* d_input2, float* d_output, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_input1, d_input2, d_output, n);
    cudaDeviceSynchronize();
}