C = gcc -Wall
LDFLAGS = -fopenmp

all: matrixMul

matrixMul: matrixMul.c
	$(C) matrixMul.c -o matrixMul.out $(LDFLAGS)

clean:
	rm -f matrixMul.out
