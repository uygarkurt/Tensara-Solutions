#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>

__global__ void huber_loss(const float* predictions, const float* targets, float* output, size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = predictions[i];
    float y = targets[i];

    float condition = abs(x - y);

    if (condition < 1){
        output[i] = pow(x - y, 2) / 2;
    } else{
        output[i] = condition - 0.5f;
    }
}

// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    huber_loss<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n);

    cudaDeviceSynchronize();
}