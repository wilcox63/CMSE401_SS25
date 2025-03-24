#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}
static long num_steps = 10000000;
__global__ void moving_average(double *series, double *avg, int steps, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < steps - range) {
        double sum = 0.0;
        for (int j = 0; j <= range; j++) {
            sum += series[i + j];
        }
        avg[i] = sum / (range + 1.0);
    }
}

int main() {
    int steps = num_steps;
    unsigned int seed = 1;
    int range = 1000;
    double *series, *avg;
    double *d_series, *d_avg;
    
    series = (double *)malloc(steps * sizeof(double));
    avg = (double *)malloc((steps - range) * sizeof(double));
    
    series[0] = 10.0;
    for (int i = 1; i < steps; i++) {
        series[i] = series[i - 1] + ((double)rand_r(&seed)) / RAND_MAX - 0.5;
    }
    
    cudaMalloc(&d_series, steps * sizeof(double));
    cudaMalloc(&d_avg, (steps - range) * sizeof(double));
    
    cudaMemcpy(d_series, series, steps * sizeof(double), cudaMemcpyHostToDevice);
    
    int blockSize = 256;
    int gridSize = (steps - range + blockSize - 1) / blockSize;
    moving_average<<<gridSize, blockSize>>>(d_series, d_avg, steps, range);
    
    cudaMemcpy(avg, d_avg, (steps - range) * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("%f %f\n\n", series[steps - 1], avg[steps - range - 1]);
    
    free(series);
    free(avg);
    cudaFree(d_series);
    cudaFree(d_avg);
    
    return 0;
}
