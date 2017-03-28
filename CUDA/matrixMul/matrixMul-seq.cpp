#include <iostream>
#include <getopt.h>
#include <random>
#include "matrix.hpp"

using namespace std;

void runProgram(int n) {
  float *matA = (float*) malloc(n * n * sizeof(float));
  float *matB = (float*) malloc(n * n * sizeof(float));
  float *matC = (float*) malloc(n * n * sizeof(float));
  
  matrix::initRandom(matA, n);
  matrix::initRandom(matB, n);
  
  matrix::seqMul(matA, matB, matC, n);
  
  free(matA);
  free(matB);
  free(matC);
}

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "Error: You must provide the size of the matrices" << endl;
    exit(EXIT_FAILURE);
  }

  int n = atoi(argv[1]);
  runProgram(n);
  
  return 0;
}

