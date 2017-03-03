CUDAC = nvcc

all: vectorAdd

vectorAdd: vectorAdd.cu
	$(CUDAC) -o vectorAdd.out vectorAdd.cu $(LDFLAGS)

clean:
	rm -f vectorAdd.out
