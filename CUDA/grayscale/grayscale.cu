#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <getopt.h>
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
    d_out_img[idx] = MAX(0, 116 * powf(y, 1.0/3.0) - 16);
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

void runProgram(Mat& image, bool show) {
  int height = image.rows;
  int width = image.cols;
  int size = width * height * sizeof(unsigned char);
  
  unsigned char *img_grayscale = (unsigned char*) malloc(size);
  unsigned char *img = (unsigned char*) image.data;

  grayscale(img, img_grayscale, width, height);

  if (show) {
    imshow("Color", Mat(height, width, CV_8UC3, img));
    waitKey(0);
    imshow("Grayscale", Mat(height, width, CV_8UC1, img_grayscale));
    waitKey(0);
  }

  free(img_grayscale);
}

void usage(char* program_name) {
  int n = 1;
  string opts[] = {"-s, --show"};
  string description[] = {
    "Show original image and result"
  };

  cout << "Usage: " << program_name << " [options ...] img1" << endl;
  cout << endl;
  cout << "Options" << endl;
  for (int i = 0; i < n; i++) {
    cout << "  " << opts[i] << ": " << description[i] << endl;
  }

  exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
  int opt, opt_index = 0;
    static struct option options[] = {
      {"show", no_argument, 0, 's'},
      {0, 0, 0, 0}
  };
  
  bool show = false;
  while ((opt = getopt_long(argc, argv, "s", options, &opt_index)) != -1) {
    switch (opt) {
      case 's':
        show = true;
        break;
      default:
        usage(argv[0]);
        break;
    }
  }
  
  if (argc - optind != 1) {
    cout << "Error: You must provide an image" << endl << endl;
    usage(argv[0]);
  }

  Mat image = imread(argv[optind]);
  if(!image.data) {
    printf("Could not open or find %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  
  runProgram(image, show);
  
  return 0;
}

