#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_normalization(const float* input, float* output, const size_t B, const size_t N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B) return;
    float sum = 0.0f;
    const float eps = 1e-5f;

    for (int j = 0; j < N; j++){
        sum += pow(input[i * N + j], 2);
    }
    float rms = sqrtf((sum / N) + eps);

    for (int j = 0; j < N; j++){
        output[i * N + j] = input[i * N + j] / rms;
    }
}

// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (B + threadsPerBlock - 1) / threadsPerBlock;
    
    rms_normalization<<<blocksPerGrid, threadsPerBlock>>>(X, Y, B, N);
    cudaDeviceSynchronize();
}