#include <cuda.h>
#include <iostream>
#include <getopt.h>
#include <random>
#include <cmath>
#include "matrix.hpp"

#define TILE_WIDTH 32

using namespace std;

void checkError(cudaError_t &err) {
  if (err != cudaSuccess) {
    printf("ERROR: %s in %s\n",cudaGetErrorString(err), __FILE__);
    exit(EXIT_FAILURE);
  }
}

__global__
void matrixMulKernel(float *matA, float *matB, float *matC, int n) {
  __shared__ float shared_matA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shared_matB[TILE_WIDTH][TILE_WIDTH];
  
  int i = blockIdx.y * TILE_WIDTH + threadIdx.y;
  int j = blockIdx.x * TILE_WIDTH + threadIdx.x;
  
  float sum = 0;
  for (int m = 0; m * TILE_WIDTH < n; m++) {
    if (i < n && m * TILE_WIDTH + threadIdx.x < n) {
      shared_matA[threadIdx.y][threadIdx.x] = matA[i * n + (m * TILE_WIDTH + threadIdx.x)];
    } else {
      shared_matA[threadIdx.y][threadIdx.x] = 0;
    }
    if (m * TILE_WIDTH + threadIdx.y < n && j < n) {
      shared_matB[threadIdx.y][threadIdx.x] = matB[(m * TILE_WIDTH + threadIdx.y) * n + j];
    } else {
      shared_matB[threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; k++) {
      sum += shared_matA[threadIdx.y][k] * shared_matB[k][threadIdx.x];
    }
    __syncthreads();
  }
    
  if (i < n && j < n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    matC[row * n + col] = sum;
  }
}

void matrixMul(float *h_matA, float *h_matB, float *h_matC, int n) {
  int size = n * n * sizeof(float);
  float *d_matA, *d_matB, *d_matC;
  cudaError_t err;
  
  err = cudaMalloc((void**) &d_matA, size); checkError(err);
  err = cudaMalloc((void**) &d_matB, size); checkError(err);
  err = cudaMalloc((void**) &d_matC, size); checkError(err);
  err = cudaMemcpy(d_matA, h_matA, size, cudaMemcpyHostToDevice); checkError(err);
  err = cudaMemcpy(d_matB, h_matB, size, cudaMemcpyHostToDevice); checkError(err);
  
  int block_size = TILE_WIDTH;
  dim3 dim_grid(ceil((double) n / block_size), ceil((double) n / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);
  matrixMulKernel<<<dim_grid, dim_block>>>(d_matA, d_matB, d_matC, n);
  cudaDeviceSynchronize();
  err = cudaMemcpy(h_matC, d_matC, size, cudaMemcpyDeviceToHost); checkError(err);
  
  err = cudaFree(d_matA); checkError(err);
  err = cudaFree(d_matB); checkError(err);
  err = cudaFree(d_matC); checkError(err);
}

float getMaxError(float *matA, float *matB, int n) {
  float max_error = -1;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float _error = fabs(matA[i * n + j] - matB[i * n + j]);
      max_error = max(max_error, _error);
    }
  }
  return max_error;
}

void runTest(float *matA, float *matB, float *d_matC, int n) {
  cout << "Finished parallel version, running sequential..." << endl;
  float *h_matC = (float*) malloc(n * n * sizeof(float));
  
  matrix::seqMul(matA, matB, h_matC, n);
  cout << "Done" << endl;
  
  float err = getMaxError(d_matC, h_matC, n);
  cout << "Max difference = " << err << endl;
}

void runProgram(int n, bool test) {
  float *matA = (float*) malloc(n * n * sizeof(float));
  float *matB = (float*) malloc(n * n * sizeof(float));
  float *matC = (float*) malloc(n * n * sizeof(float));
  
  matrix::initRandom(matA, n);
  matrix::initRandom(matB, n);
  
  matrixMul(matA, matB, matC, n);
  
  if (test) {
    runTest(matA, matB, matC, n);
  }
  
  free(matA);
  free(matB);
  free(matC);
}

void usage(char* program_name) {
  int n = 1;
  string opts[] = {"-t, --test"};
  string description[] = {
    "Test against sequential version"
  };

  cout << "Usage: " << program_name << " [options ...] num" << endl;
  cout << endl;
  cout << "Options" << endl;
  for (int i = 0; i < n; i++) {
    cout << "  " << opts[i] << ": " << description[i] << endl;
  }

  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  int opt, opt_index = 0;
    static struct option options[] = {
      {"test", no_argument, 0, 't'},
      {0, 0, 0, 0}
  };
  
  bool test = false;
  while ((opt = getopt_long(argc, argv, "t", options, &opt_index)) != -1) {
    switch (opt) {
      case 't':
        test = true;
        break;
      default:
        usage(argv[0]);
        break;
    }
  }
  
  if (argc - optind != 1) {
    cout << "Error: You must provide the size of the matrices" << endl << endl;
    usage(argv[0]);
  }

  int n = atoi(argv[optind]);
  runProgram(n, test);
  
  return 0;
}

