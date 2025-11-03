#include <cuda_runtime.h>

__global__ void average_pooling(const float* input, const int kernel_size, const int stride, const int padding, float* output, const size_t H_out, const size_t H){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H_out) return;
    float sum = 0.0f;

    for (int m = 0; m < kernel_size; m++){
        int idx = stride * i + m - padding;

        if (idx >= 0 && idx < H){
            sum += input[idx];
        }
    }
    output[i] = sum / kernel_size;
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {
    size_t H_out = ((H + 2 * padding - kernel_size) / stride) + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (H_out + threadsPerBlock - 1) / threadsPerBlock;

    average_pooling<<<blocksPerGrid, threadsPerBlock>>>(input, kernel_size, stride, padding, output, H_out, H);
    cudaDeviceSynchronize();
}