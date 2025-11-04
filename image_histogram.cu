#include <cuda_runtime.h>

__global__ void image_histogram(const float* image, const int num_bins, float* histogram, const size_t height, const size_t width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >=height || col >= width) return;

    int idx = row * width + col;
    int pixel = static_cast<int>(image[idx]);
    
    atomicAdd(&histogram[pixel], 1.0f); // Race Condition
}

// Note: image, histogram are all device pointers to float32 arrays
extern "C" void solution(const float* image, int num_bins, float* histogram, size_t height, size_t width) {
    cudaMemset(histogram, 0, num_bins * sizeof(float));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
					    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    image_histogram<<<blocksPerGrid, threadsPerBlock>>>(image, num_bins, histogram, height, width);

    cudaDeviceSynchronize();
}