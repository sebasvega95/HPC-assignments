CUDAC = nvcc
CC = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

all: parallel sequential cv-sequential cv-parallel memory

parallel: sobel.cu
	$(CUDAC) $(CFLAGS) sobel.cu $(LIBS) -o sobel.out
	
sequential: sobel-seq.cpp
	$(CC) $(CFLAGS) sobel-seq.cpp $(LIBS) -o sobel-seq.out
	
cv-sequential: cv-sobel-seq.cpp
	$(CC) $(CFLAGS) cv-sobel-seq.cpp $(LIBS) -o cv-sobel-seq.out

cv-parallel: cv-sobel.cpp
	$(CC) $(CFLAGS) cv-sobel.cpp $(LIBS) -o cv-sobel.out

memory: sobel-mem.cu
	$(CUDAC) $(CFLAGS) sobel-mem.cu $(LIBS) -o sobel-mem.out
	
clean:
	rm *.out

