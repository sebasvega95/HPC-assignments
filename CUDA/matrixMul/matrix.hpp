#ifndef MULT_HPP
#define MULT_HPP

#include <iostream>
#include <chrono>

using namespace std;

namespace matrix
{
  void initRandom(float *mat, int n) {
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_real_distribution<float> dist(0.0, 1.0);
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        mat[i * n + j] = dist(gen);
      }
    }
  }

  void seqMul(float *matA, float *matB, float *matC, int n) {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
          sum += matA[i * n + k] * matB[k * n + j];
        }
        matC[i * n + j] = sum;
      }
    }
  }
  
  void print(string s, float *mat, int n) {
    cout << s << " = [" << endl;
    for (int i = 0; i < n; i++) {
      cout << " ";
      for (int j = 0; j < n; j++) {
        cout << " " << mat[i * n + j];
      }
      cout << " ;" << endl;
    }
    cout << "]" << endl;
  }
}

#endif
