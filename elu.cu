#include <cuda_runtime.h>
#include <math.h>

__global__ void elu(const float* input, float* output, const size_t n, const size_t m, const float alpha){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= m) return;

    float x = input[row * m + col];
    if (x > 0){
        output[row * m + col] = x;
    } else{
        output[row * m + col] = alpha * (exp(x) - 1);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
					    (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    
    elu<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m, alpha);
    cudaDeviceSynchronize();
}