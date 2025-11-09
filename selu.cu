#include <cuda_runtime.h>
#include <math.h>

__global__ void selu(const float* input, float* output, const size_t n, const size_t m){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= n || col >= m) return;

    float x = input[row * m + col];

    float alpha = 1.67326f;
    float lambda = 1.0507f;
    float el1 = 0.0f;
    float el2 = 0.0f;

    if (el1 < x){
        el1 = x;
    }

    float t_calc = alpha * (exp(x) - 1);
    if (el2 > t_calc){
        el2 = t_calc;
    }

    output[row * m + col] = lambda * (el1 + el2);
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {
    dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(n + threadsPerBlock.y - 1) / threadsPerBlock.y);

	selu<<<blocksPerGrid, threadsPerBlock>>>(input, output, n, m);
	cudaDeviceSynchronize();
}