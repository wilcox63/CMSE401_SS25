#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include <assert.h>
#include "png_util.h"
#define MAX_N 20000
#define CUDA_CALL(x) {cudaError_t cuda_error__ = (x); if (cuda_error__) printf("CUDA error: " #x " returned \"%s\"\n", cudaGetErrorString(cuda_error__));}


char plate[2][(MAX_N + 2) * (MAX_N + 2)];
int which = 0;
int n;

unsigned char *d_plate[2]; // For GPU plates

__global__ void update_plate(unsigned char* d_plate_current, unsigned char* d_plate_next, int n, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i <= n && j > 0 && j <= n) {
        int index = i * (n + 2) + j;
        int num = d_plate_current[index - n - 3] 
                + d_plate_current[index - n - 2]
                + d_plate_current[index - n - 1]
                + d_plate_current[index - 1]
                + d_plate_current[index + 1]
                + d_plate_current[index + n + 1]
                + d_plate_current[index + n + 2]
                + d_plate_current[index + n + 3];
        if (d_plate_current[index]) {
            d_plate_next[index] = (num == 2 || num == 3) ? 1 : 0;
        } else {
            d_plate_next[index] = (num == 3);
        }
    }
}

void copy_to_gpu() {
    CUDA_CALL(cudaMemcpy(d_plate[which], plate[which], sizeof(char) * (n + 2) * (n + 2), cudaMemcpyHostToDevice));
}

void copy_from_gpu() {
    CUDA_CALL(cudaMemcpy(plate[which], d_plate[which], sizeof(char) * (n + 2) * (n + 2), cudaMemcpyDeviceToHost));
}

void allocate_gpu_memory() {
    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaMalloc((void**)&d_plate[i], sizeof(char) * (n + 2) * (n + 2)));
    }
}

void free_gpu_memory() {
    for (int i = 0; i < 2; i++) {
        CUDA_CALL(cudaFree(d_plate[i]));
    }
}

void iteration() {
    dim3 blockSize(16, 16); // Adjust block size for better performance
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // Run the kernel on the GPU
    update_plate<<<gridSize, blockSize>>>(d_plate[which], d_plate[!which], n, which);
    CUDA_CALL(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Switch between plates
    which = !which;
}

void print_plate() {
    if (n < 60) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                printf("%d", (int)plate[which][i * (n + 2) + j]);
            }
            printf("\n");
        }
    } else {
        printf("Plate too large to print to screen\n");
    }
    printf("\0");
}

void plate2png(char* filename) {
    unsigned char *img = (unsigned char*)malloc(n * n * sizeof(char));
    image_size_t sz;
    sz.width = n;
    sz.height = n;

    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            int pindex = i * (n + 2) + j;
            int index = (i - 1) * (n) + j;
            if (plate[!which][pindex] > 0)
                img[index] = 255;
            else
                img[index] = 0;
        }
    }
    printf("Writing file\n");
    write_png_file(filename, img, sz);
    printf("done writing png\n");
    free(img);
    printf("done freeing memory\n");
}

int main() {
    int M;
    char line[MAX_N];

    if (scanf("%d %d", &n, &M) == 2) {
        if (n > 0) {
            memset(plate[0], 0, sizeof(char) * (n + 2) * (n + 2));
            memset(plate[1], 0, sizeof(char) * (n + 2) * (n + 2));
            for (int i = 1; i <= n; i++) {
                scanf("%s", &line);
                for (int j = 0; j < n; j++) {
                    plate[0][i * (n + 2) + j + 1] = line[j] - '0';
                }
            }
        } else {
            n = MAX_N;
            for (int i = 1; i <= n; i++)
                for (int j = 0; j < n; j++)
                    plate[0][i * (n + 2) + j + 1] = (char)rand() % 2;
        }

        allocate_gpu_memory();

        copy_to_gpu();

        for (int i = 0; i < M; i++) {
            printf("\nIteration %d:\n", i);
            print_plate();
            iteration(); // Perform iteration on GPU
        }

        copy_from_gpu();

        printf("\n\nFinal:\n");
        plate2png("plate.png");
        print_plate();

        free_gpu_memory();
    }

    return 0;
}
