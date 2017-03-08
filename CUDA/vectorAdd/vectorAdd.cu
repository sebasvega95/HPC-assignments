#include <cuda.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define max(x, y) x > y ? x : y
#define TIME(f, msg) \
  _begin = clock(); \
  (f); \
  _end = clock(); \
  printf("%s done in %f\n", (msg), (float)(_end - _begin) / CLOCKS_PER_SEC);

void testRand(int *a, int n, int max) {
  for (int i = 0; i < n; i++) {
    a[i] = rand() % max;
  }
}

void print(char *pref, int *a, int n) {
  printf("%s", pref);
  for (int i = 0; i < n; i++)
    printf("%d ", a[i]);
  printf("\n");
}

int getMaxError(int *x, int *y, int n) {
  int max_error = -INT_MAX;
  for (int i = 0; i < n; i++) {
    max_error = max(max_error, abs(x[i] - y[i]));
  }
  return max_error;
}

void vectorAddSeq(int* h_a, int *h_b, int *h_c, int n) {
  for (int i = 0; i < n; i++) {
    h_c[i] = h_a[i] + h_b[i];
  }
}

__global__
void vectorAddKernel(int* d_a, int *d_b, int *d_c, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n) {
    d_c[i] = d_a[i] + d_b[i];
  }
}

void checkError(cudaError_t &err) {
  if (err != cudaSuccess) {
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

void vectorAdd(int* h_a, int *h_b, int *h_c, int n, int num_thread) {
  int size = n * sizeof(int);
  int *d_a, *d_b, *d_c;
  cudaError_t err;
  
  err = cudaMalloc((void **) &d_a, size); checkError(err);
  err = cudaMalloc((void **) &d_b, size); checkError(err);
  err = cudaMalloc((void **) &d_c, size); checkError(err);
  
  err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice); checkError(err);
  err = cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice); checkError(err);
  
  int num_blocks = ceil((double)n / num_thread);
  vectorAddKernel<<<num_blocks, num_thread>>>(d_a, d_b, d_c, n);
  err = cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost); checkError(err);
  
  cudaDeviceSynchronize();
  
  err = cudaFree(d_a); checkError(err);
  err = cudaFree(d_b); checkError(err);
  err = cudaFree(d_c); checkError(err);
}

int main(int argc, char const *argv[]) {
  int n = 10000000;
  int num_threads = 256;
  int *a, *b, *c_seq, *c_par;
  clock_t _begin, _end;
  
  a = (int *) malloc(n * sizeof(int));
  b = (int *) malloc(n * sizeof(int));
  c_seq = (int *) malloc(n * sizeof(int));
  c_par = (int *) malloc(n * sizeof(int));

  testRand(a, n, 10);
  testRand(b, n, 10);

  TIME(vectorAddSeq(a, b, c_seq, n), "Sequential");
  TIME(vectorAdd(a, b, c_par, n, num_threads), "Parallel");
  
  if (n < 20) {
    print("Seq ", c_seq, n);
    print("Par ", c_par, n);
  }
  printf("Max error: %d\n", getMaxError(c_seq, c_par, n));

  free(a);
  free(b);
  free(c_seq);
  free(c_par);

  return 0;
}
