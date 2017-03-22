CUDAC = nvcc
CC = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: parallel sequential

parallel: grayscale.cu
	$(CUDAC) $(CFLAGS) grayscale.cu $(LIBS) -o grayscale.out
	
sequential: grayscale-seq.cpp
	$(CC) $(CFLAGS) grayscale-seq.cpp $(LIBS) -o grayscale-seq.out

