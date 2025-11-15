// YouTube Tutorial: https://www.youtube.com/watch?v=zbmg-K7dJUE

#include <cuda_runtime.h>

__global__ void hinge_loss(const float* predictions, const float* targets, float* output, const size_t n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float max = 0.0f;
    float t_calc = 1 - predictions[i] * targets[i];

    if (t_calc > max){
        max = t_calc;
    }
    
    output[i] = max;
}

// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    hinge_loss<<<blocksPerGrid, threadsPerBlock>>>(predictions, targets, output, n);
    cudaDeviceSynchronize();
}