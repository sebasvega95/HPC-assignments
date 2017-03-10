CUDAC = nvcc
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

times2Filter: times2Filter.cu
	$(CUDAC) $(CFLAGS) times2Filter.cu $(LIBS) -o times2Filter.out
