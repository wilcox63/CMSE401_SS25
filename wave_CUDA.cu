#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NX 500
#define NT 100000

__global__ void computeDvdt(double *y, double *dvdt, double dx2inv, int nx) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i < nx-1){
		dvdt[i] = (y[i+1] + y[i-1] - 2.0 * y[i])*dx2inv;
	}
}

__global__ void updateVy(double *y, double *v,  double *dvdt, double dt, int nx) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i < nx-1){
		v[i] = v[i] + dt * dvdt[i];
		y[i] = y[i] + dt * v[i];
	}
}

int main(){
	int i;
	double *x, *y, *v, *dvdt;
	double *d_y, *d_v, *d_dvdt;
	double dt, dx, dx2inv;
	double max = 10.0, min = 0.0;

	x = (double*)malloc(NX * sizeof(double));
	y = (double*)malloc(NX * sizeof(double));
	v = (double*)malloc(NX * sizeof(double));
	dvdt = (double*)malloc(NX * sizeof(double));

	dx = (max - min) / (double)(NX);
	for (i = 0; i < NX ; i++){
		x[i] = min + i * dx;
		y[i] = exp(-(x[i] - 5.0) * x[i] - 5.0);
		v[i] = 0.0;
		dvdt[i] = 0.0;
	}
	dt = 10.0 / (double)(NT);
	dx2inv = 1.0 / (dx * dx);

	cudaMalloc(&d_y, NX * sizeof(double));
	cudaMalloc(&d_v, NX * sizeof(double));
	cudaMalloc(&d_dvdt, NX * sizeof(double));

	cudaMemcpy(d_y, y, NX * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_v, v, NX * sizeof(double), cudaMemcpyHostToDevice);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int threadConfigs[] = {64, 128, 256, 512};
	int numConfigs = sizeof(threadConfigs) / sizeof(threadConfigs[0]);
	float prevTime = 0.0;

	for (int config = 0; config < numConfigs; config++) {
		int ThreadsPerBlock = threadConfigs[config];
		int BlocksPerGrid = (NX + ThreadsPerBlock - 1) / ThreadsPerBlock;

		cudaEventRecord(start);

		for (int it = 0; it < NT - 1; it ++){
			computeDvdt<<<BlocksPerGrid, ThreadsPerBlock>>>(d_y,d_dvdt, dx2inv, NX);
			updateVy<<<BlocksPerGrid, ThreadsPerBlock>>>(d_y,d_v, d_dvdt, dt, NX);
		}

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float ms;
		cudaEventElapsedTime(&ms, start, stop);

		float speedup = (config == 0) ? 1.0 : (prevTime / ms);

		printf( "%d, %f, %f\n", ThreadsPerBlock, ms, speedup);

		if (config == 0) prevTime = ms;
	}

	cudaMemcpy(y, d_y, NX * sizeof(double), cudaMemcpyDeviceToHost);

	for (i = NX/2 - 10; i < NX/2 + 10; i++){
		printf("%g %g\n", x[i], y[i]);
	}

	free(x); free(y); free(v); free(dvdt);
	cudaFree(d_y); cudaFree(d_v); cudaFree(d_dvdt);

	return 0;
}
