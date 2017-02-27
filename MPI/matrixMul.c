#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define N 7
#define M 7
#define L 7
#define MUL_MAT_TAG 1

void matrixVectorMul(int *A, int *x, int *y, int n, int m) {
  for (int i = 0; i < n; i++) {
    y[i] = 0;
    for (int j = 0; j < m; j++) {
      y[i] += A[i * m + j] * x[j];
    }
  }
}

void transpose(int *A, int *At, int n, int m) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      At[j * n + i] = A[i * m + j];
    }
  }
}

void testRand(int *A, int n, int m, int max) {
  int size = n * m;

  for (int i = 0; i < size; i++) {
    A[i] = rand() % max;
  }
}

void print(char *pref, int *A, int n, int m) {
  printf("%s[\n", pref);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf(" %d", A[i * m + j]);
    }
    if (i == n - 1) {
      printf(" ]\n");
    } else {
      printf(" ;\n");
    }
  }
  
}

int main(int argc, char const *argv[]) {
  int np, pid;
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  
  if (np < 2) {
    printf("I require at least two processors\n");
    MPI_Finalize();
    return 0;
  }
  
  int *A, *B, *C;
  int num_cols = L / (np - 1);
  if (pid == 0) {
    A = (int *) malloc(N * M * sizeof(int));
    B = (int *) malloc(M * L * sizeof(int));
    C = (int *) malloc(N * L * sizeof(int));
    
    testRand(A, N, M, 10);
    testRand(B, M, L, 10);
    
    print("A = ", A, N, M);
    print("B = ", B, M, L);
    
    int *Bt = (int *) malloc(L * M * sizeof(int));
    int *Ct = (int *) malloc(L * N * sizeof(int));
    transpose(B, Bt, M, L);
    
    for (int i = 1; i < np; i++) {
      MPI_Send(A, N * L, MPI_INT, i, MUL_MAT_TAG, MPI_COMM_WORLD);
      MPI_Send(Bt + M * (i - 1) * num_cols, num_cols * M, MPI_INT, i, MUL_MAT_TAG, MPI_COMM_WORLD);
    }
    
    for (int i = 1; i < np; i++) {
      MPI_Recv(Ct + N * (i - 1) * num_cols, num_cols * N, MPI_INT, i, MUL_MAT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    int missing = N % (np - 1);
    int missing_offset = N - missing;
    
    for (int i = 0; i < missing; i++) {
      matrixVectorMul(A, Bt + (missing_offset + i) * M, Ct + (missing_offset + i) * N, N, M);
    }
    
    transpose(Ct, C, L, N);
    print("C = ", C, N, L);

    free(Bt);
    free(Ct);
  } else {
    A = (int *) malloc(N * M * sizeof(int));
    B = (int *) malloc(num_cols * M * sizeof(int));
    C = (int *) malloc(num_cols * N * sizeof(int));
    
    MPI_Recv(A, N * M, MPI_INT, 0, MUL_MAT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(B, M * num_cols, MPI_INT, 0, MUL_MAT_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    for (int i = 0; i < num_cols; i++) {
      matrixVectorMul(A, B + i * M, C + i * N, N, M);
    }
    
    MPI_Send(C, num_cols * N, MPI_INT, 0, MUL_MAT_TAG, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  free(A);
  free(B);
  free(C);
  
  return 0;
}

