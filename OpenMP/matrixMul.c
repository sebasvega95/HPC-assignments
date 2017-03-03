#include <limits.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 1000
#define M 1000
#define L 1000
#define TIME(f, msg) _begin = omp_get_wtime(); (f); _end = omp_get_wtime(); printf("%s done in %f\n", (msg), _end - _begin);

void matrixMulSeq(int *A, int *B, int *C, int n, int m, int l) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      C[i * m + j] = 0;
      for (int k = 0; k < l; k++)
        C[i * m + j] += A[i * l + k] * B[k * m + j];
    }
  }
}

void matrixMulMp(int *A, int *B, int *C, int n, int m, int l) {
  int i, j, k;

  #pragma omp parallel shared(A, B, C, n, m) private(i, j, k)
  {
    #pragma omp for schedule(static)
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        C[i * m + j] = 0;
        for (k = 0; k < l; k++) {
          C[i * m + j] += A[i * l + k] * B[k * m + j];
        }
      }
    }
  }
}

void testRand(int *A, int *B, int n, int m, int l, int max) {
  int sizeA = n * l;
  int sizeB = l * m;
  int _A[sizeA], _B[sizeB];

  for (int i = 0; i < sizeA; i++) {
    _A[i] = rand() % max;
  }
  for (int i = 0; i < sizeB; i++) {
    _B[i] = rand() % max;
  }

  memcpy(A, _A, sizeA * sizeof(int));
  memcpy(B, _B, sizeB * sizeof(int));
}

int getMaxError(int *X, int *Y, int n, int m) {
  int max_error = -INT_MAX;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      int _error = abs(X[i * n + j] - Y[i * n + j]);
      max_error = _error > max_error ? _error : max_error;
    }
  }
  return max_error;
}

int main(int argc, char *argv[]) {
  int n = N, m = M, l = L;
  int *A = (int*) malloc(n * l * sizeof(int));
  int *B = (int*) malloc(l * m * sizeof(int));
  int *C = (int*) malloc(n * m * sizeof(int));
  int *Cseq = (int*) malloc(n * m * sizeof(int));
  double _begin, _end;

  TIME(testRand(A, B, n, m, l, 50), "Init");
  TIME(matrixMulSeq(A, B, Cseq, n, m, l), "Seq ");
  TIME(matrixMulMp(A, B, C, n, m, l), "MP  ");

  printf("Max error: %d\n", getMaxError(C, Cseq, n, m));

  free(A);
  free(B);
  free(C);

  return 0;
}
