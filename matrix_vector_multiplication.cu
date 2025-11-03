#include <cuda_runtime.h>

__global__ void mat_vec_dot(const float* input_a, const float* input_b, float* output_c, const size_t m, const size_t k){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row > m) return;
    float sum = 0.0f;

    for (int i = 0; i < k; i++){
        sum += input_a[row * k + i] * input_b[i];
    }
    output_c[row] = sum;
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t m, size_t k) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (m + threadsPerBlock - 1) / threadsPerBlock;

    mat_vec_dot<<<blocksPerGrid, threadsPerBlock>>>(input_a, input_b, output_c, m, k);
    cudaDeviceSynchronize();
}
