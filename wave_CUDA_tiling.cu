#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCK_SIZE = 256

__global__ void update_dvdt(double* dvdt, double* y, double dx2inv, int nx){
    __shared__ double y_shared[BLOCK_SIZE + 2];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int local_idx = threadIdx.x + 1;
    
    if (i >= nx - 2) return;
    
   
    y_shared[local_idx] = y[i];
    if (threadIdx.x == 0 && i > 0)
        y_shared[0] = y[i - 1];
    if (threadIdx.x == blockDim.x - 1 && i < nx - 1)
        y_shared[local_idx + 1] = y[i + 1];
    
    __syncthreads();
    
   
    dvdt[i] = (y_shared[local_idx + 1] + y_shared[local_idx - 1] - 2.0 * y_shared[local_idx]) * dx2inv;
}

__global__ void update_vy(double* v, double* y, double dt, double* dvdt, int nx){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i >= nx - 2) return;
    
    v[i] = v[i] + dt * dvdt[i];
    y[i] = y[i] + dt * v[i];
}

int main() {
    int nx = 500;
    int nt = 1000;
    int i, it;
    double dx, dt, dx2inv;
    double *y, *v, *dvdt;
    double *d_y, *d_v, *d_dvdt;
    
    y = (double*)malloc(nx * sizeof(double));
    v = (double*)malloc(nx * sizeof(double));
    dvdt = (double*)malloc(nx * sizeof(double));
    
    cudaMalloc(&d_y, nx * sizeof(double));
    cudaMalloc(&d_v, nx * sizeof(double));
    cudaMalloc(&d_dvdt, nx * sizeof(double));
    
    dx = 10.0 / (nx - 1);
    dt = 10.0 / nt;
    dx2inv = 1.0 / (dx * dx);
    
    for (i = 0; i < nx; i++) {
        y[i] = exp(-pow((i * dx - 5.0), 2));
        v[i] = 0.0;
        dvdt[i] = 0.0;
    }
    
    cudaMemcpy(d_y, y, nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, nx * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dvdt, dvdt, nx * sizeof(double), cudaMemcpyHostToDevice);
    
    int blocks = (nx + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks < 10) blocks = 10;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (it = 0; it < nt - 1; it++) {
        update_dvdt<<<blocks, BLOCK_SIZE>>>(d_dvdt, d_y, dx2inv, nx);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
        update_vy<<<blocks, BLOCK_SIZE>>>(d_v, d_y, dt, d_dvdt, nx);
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
            return -1;
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution Time: %f ms\n", milliseconds);
    
    cudaMemcpy(y, d_y, nx * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (i = nx / 2; i < nx / 2 + 10; i++) {
        printf("%f, ", y[i]);
    }
    
    cudaFree(d_y);
    cudaFree(d_v);
    cudaFree(d_dvdt);
    free(y);
    free(v);
    free(dvdt);
    
    return 0;
}
