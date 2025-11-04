#include <cuda_runtime.h>

__global__ void matrix_scalar_multiplication(const float* input_matrix, const float scalar, float* output_matrix, const size_t n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;

    output_matrix[row * n + col] = input_matrix[row * n + col] * scalar;
}

// Note: input_matrix, output_matrix are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const float scalar, float* output_matrix, size_t n) {
    dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
						(n + threadsPerBlock.y - 1) / threadsPerBlock.y);

	matrix_scalar_multiplication<<<blocksPerGrid, threadsPerBlock>>>(input_matrix, scalar, output_matrix, n);
	cudaDeviceSynchronize();
}