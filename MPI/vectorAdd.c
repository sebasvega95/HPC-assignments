#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#define N 21
#define ADD_ARRAY_TAG 1

void add(int* a, int *b, int *c, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

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
  
  int *a, *b, *c;
  int chunk_size = N / (np - 1);
  if (pid == 0) {
    a = (int *) malloc(N * sizeof(int));
    b = (int *) malloc(N * sizeof(int));
    c = (int *) malloc(N * sizeof(int));
    
    testRand(a, N, 10);
    testRand(b, N, 10);
    
    print("a = ", a, N);
    print("b = ", b, N);
    
    for (int i = 1; i < np; i++) {
      MPI_Send(a + (i - 1) * chunk_size, chunk_size, MPI_INT, i, ADD_ARRAY_TAG, MPI_COMM_WORLD);
      MPI_Send(b + (i - 1) * chunk_size, chunk_size, MPI_INT, i, ADD_ARRAY_TAG, MPI_COMM_WORLD);
    }
    
    for (int i = 1; i < np; i++) {
      MPI_Recv(c + (i - 1) * chunk_size, chunk_size, MPI_INT, i, ADD_ARRAY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    int missing = N % (np - 1);
    int missing_offset = N - missing;
    add(a + missing_offset, b + missing_offset, c + missing_offset, missing);
    
    print("c = ", c, N);
  } else {
    a = (int *) malloc(chunk_size * sizeof(int));
    b = (int *) malloc(chunk_size * sizeof(int));
    c = (int *) malloc(chunk_size * sizeof(int));
    
    MPI_Recv(a, chunk_size, MPI_INT, 0, ADD_ARRAY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(b, chunk_size, MPI_INT, 0, ADD_ARRAY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    add(a, b, c, chunk_size);
    MPI_Send(c, chunk_size, MPI_INT, 0, ADD_ARRAY_TAG, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  free(a);
  free(b);
  free(c);
  
  return 0;
}

