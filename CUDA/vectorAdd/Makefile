CUDAC = nvcc

all: vectorAdd

vectorAdd: vectorAdd.cu
	$(CUDAC) -o vectorAdd.out vectorAdd.cu

clean:
	rm -f vectorAdd.out
