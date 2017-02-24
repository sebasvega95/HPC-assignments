#include <stdio.h>
#include <stdlib.h>
#define N 10000000

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

void print(int *a, int n) {
  for (int i = 0; i < n; i++)
    printf("%d ", a[i]);
  printf("\n");
}

int main(int argc, char const *argv[]) {
  int *a, *b, *c;
  a = (int *) malloc(N * sizeof(int));
  b = (int *) malloc(N * sizeof(int));
  c = (int *) malloc(N * sizeof(int));

  testRand(a, N, 10);
  testRand(b, N, 10);

  add(a, b, c, N);

  // printf("a = "); print(a, N);
  // printf("b = "); print(b, N);
  // printf("c = "); print(c, N);

  free(a);
  free(b);
  free(c);

  return 0;
}
