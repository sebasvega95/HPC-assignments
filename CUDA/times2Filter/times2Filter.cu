#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <cstdio>

using namespace cv;
using namespace std;

void checkError(cudaError_t &err) {
  if (err != cudaSuccess) {
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);
    exit(EXIT_FAILURE);
  }
}

__global__
void times2FilterKernel(unsigned char* d_img, unsigned char* d_out_img, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = row * width + col;
  
  if (row < height && col < width) {
    d_out_img[idx] = 2 * d_img[idx];
  }
}

void times2Filter(unsigned char *h_img, unsigned char *img_times2, int width, int height) {
  int size = width * height * sizeof(unsigned char);
  unsigned char *d_img, *d_out_img;
  cudaError_t err;
  
  err = cudaMalloc((void**) &d_img, size); checkError(err);
  err = cudaMalloc((void**) &d_out_img, size); checkError(err);
  err = cudaMemcpy(d_img, h_img, size, cudaMemcpyHostToDevice); checkError(err);
  
  int block_size = 32;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);
  times2FilterKernel<<<dim_grid, dim_block>>>(d_img, d_out_img, width, height);
  cudaDeviceSynchronize();
  err = cudaMemcpy(img_times2, d_out_img, size, cudaMemcpyDeviceToHost); checkError(err);
  
  err = cudaFree(d_img); checkError(err);
  err = cudaFree(d_out_img); checkError(err);
}

void showSideBySide(Mat &img1, Mat &img2, int type, const char *str) {
  Size sz1 = img1.size();
  Size sz2 = img2.size();
  
  Mat img3(sz1.height, sz1.width + sz2.width, type);
  Mat left(img3, Rect(0, 0, sz1.width, sz1.height));
  img1.copyTo(left);
  Mat right(img3, Rect(sz1.width, 0, sz2.width, sz2.height));
  img2.copyTo(right);
  
  imshow(str, img3);
  waitKey(0);
}

int main(int argc, char** argv) {
  if(argc != 2) {
    printf("Usage: %s <image>\n", argv[0]);
    return -1;
  }

  Mat image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

  if(!image.data) {
    printf("Could not open or find the %s\n", argv[1]);
    return -1;
  }
  
  int height = image.rows;
  int width = image.cols;
  int size = width * height * sizeof(unsigned char);
  
  unsigned char *img_times2 = (unsigned char*) malloc(size);
  unsigned char *img = (unsigned char*) image.data;

  times2Filter(img, img_times2, width, height);

  Mat image_output(height, width, CV_8UC1, (void*) img_times2);
  showSideBySide(image, image_output, CV_8UC1, "Result");

  free(img_times2);
  
  return 0;
}
