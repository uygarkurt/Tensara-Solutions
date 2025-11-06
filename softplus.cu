#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus(const float* input, float* output, const size_t n, const size_t m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= m) return;

    output[row * m + col] = log(1.0 + exp(input[row * m + col]));
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
					    (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    softplus<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
    cudaDeviceSynchronize();
}