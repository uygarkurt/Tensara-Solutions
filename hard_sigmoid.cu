#include <cuda_runtime.h>

__global__ void hard_sigmoid(const float* input, float* output, const size_t n, const size_t m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= n || col >= m) return;

    float x = input[row * m + col];

    if (x <= -3){
        output[row * m + col] = 0;
    } 
    else if (x >= 3){
        output[row * m + col] = 1;
    } 
    else{
        output[row * m + col] = (x + 3) / 6;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
					    (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    hard_sigmoid<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
    cudaDeviceSynchronize();
}