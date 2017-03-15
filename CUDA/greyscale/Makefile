CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

times2Filter: grayscale.cu
	$(CUDAC) $(CFLAGS) grayscale.cu $(LIBS) -o grayscale.out
