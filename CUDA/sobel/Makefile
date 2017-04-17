CUDAC = nvcc
CC = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: parallel sequential cv-sequential cv-parallel

sequential: sobel-seq.cpp
	$(CC) $(CFLAGS) sobel-seq.cpp $(LIBS) -o sobel-seq.out
	
cv-sequential: cv-sobel-seq.cpp
	$(CC) $(CFLAGS) cv-sobel-seq.cpp $(LIBS) -o cv-sobel-seq.out

