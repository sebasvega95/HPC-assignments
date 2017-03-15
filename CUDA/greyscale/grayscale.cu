#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <cstdio>

#define BLUE 0
#define GREEN 1
#define RED 2

#define CHANNELS 3
#define GAMMA 2.2

using namespace cv;
using namespace std;

void checkError(cudaError_t &err) {
  if (err != cudaSuccess) {
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__
void grayscaleKernel(unsigned char* d_img, unsigned char* d_out_img, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = row * width + col;
  
  if (row < height && col < width) {
    float b = d_img[idx * CHANNELS + BLUE];
    float g = d_img[idx * CHANNELS + GREEN];
    float r = d_img[idx * CHANNELS + RED];
    float y = 0.2126 * powf(r / 255, GAMMA) + 0.7152 * powf(g / 255, GAMMA) + 0.0722 * powf(b / 255, GAMMA);
    d_out_img[idx] = 116 * powf(y, 1.0/3.0) - 16;
  }
}

void grayscale(unsigned char *h_img, unsigned char *img_grayscale, int width, int height) {
  int size = width * height * sizeof(unsigned char);
  unsigned char *d_img, *d_out_img;
  cudaError_t err;
  
  err = cudaMalloc((void**) &d_img, 3 * size); checkError(err);
  err = cudaMalloc((void**) &d_out_img, size); checkError(err);
  err = cudaMemcpy(d_img, h_img, 3 * size, cudaMemcpyHostToDevice); checkError(err);
  
  int block_size = 32;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);
  grayscaleKernel<<<dim_grid, dim_block>>>(d_img, d_out_img, width, height);
  cudaDeviceSynchronize();
  err = cudaMemcpy(img_grayscale, d_out_img, size, cudaMemcpyDeviceToHost); checkError(err);
  
  err = cudaFree(d_img); checkError(err);
  err = cudaFree(d_out_img); checkError(err);
}

int main(int argc, char** argv) {
  if(argc != 2) {
    printf("Usage: %s <image>\n", argv[0]);
    return -1;
  }

  Mat image = imread(argv[1]);

  if(!image.data) {
    printf("Could not open or find %s\n", argv[1]);
    return -1;
  }
  
  int height = image.rows;
  int width = image.cols;
  int size = width * height * sizeof(unsigned char);
  
  unsigned char *img_grayscale = (unsigned char*) malloc(size);
  unsigned char *img = (unsigned char*) image.data;

  grayscale(img, img_grayscale, width, height);

  imshow("Color", Mat(height, width, CV_8UC3, img));
  waitKey(0);
  imshow("Grayscale", Mat(height, width, CV_8UC1, img_grayscale));
  waitKey(0);

  free(img_grayscale);
  
  return 0;
}
