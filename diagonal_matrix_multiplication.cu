#include <cuda_runtime.h>

__global__ void diagonal_matrix_multiplication(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    for (int j = 0; j < m; j++){
        output_c[i * m + j] = diagonal_a[i] * input_b[i * m + j];
    }
}

// Note: diagonal_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* diagonal_a, const float* input_b, float* output_c, size_t n, size_t m) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    diagonal_matrix_multiplication<<<blocksPerGrid, threadsPerBlock>>>(diagonal_a, input_b, output_c, n, m);
	cudaDeviceSynchronize();
}