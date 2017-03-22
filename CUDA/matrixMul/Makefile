CUDAC = nvcc
CC = g++
CFLAGS = -std=c++11

all: parallel sequential

parallel: matrixMul.cu
	$(CUDAC) $(CFLAGS) matrixMul.cu -o matrixMul.out
	
sequential: matrixMul-seq.cpp
	$(CC) $(CFLAGS) matrixMul-seq.cpp -o matrixMul-seq.out

