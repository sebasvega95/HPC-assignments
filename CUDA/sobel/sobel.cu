#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdio>
#define checkError(err)                                                                \
  if ((err) != cudaSuccess) {                                                          \
    printf("ERROR: %s in %s, line %d\n",cudaGetErrorString(err), __FILE__, __LINE__);  \
    exit(EXIT_FAILURE);                                                                \
  }

using namespace cv;
using namespace std;

__device__
bool inside_image(int row, int col, int width, int height) {
  return row >= 0 && row < height && col >= 0 && col < width;
}

__global__
void convolutionKernel(unsigned char* image, float* kernel, float* out_image, int kernel_n, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < height && col < width) {
    int n = kernel_n / 2;
    float accumulation = 0;
    for (int i = -n; i <= n; i++) {
      for (int j = -n; j <= n; j++) {
        if (inside_image(row + i, col + j, width, height)) {
          int image_idx = (row + i) * width + (col + j);
          int kernel_idx = (n + i) * kernel_n + (n + j);
          accumulation += image[image_idx] * kernel[kernel_idx];
        }
      }
    }
    out_image[row * width + col] = accumulation;
  }
}

__device__
float Q_rsqrt( float number ) {
  // Implementation in 1999 in the source code of Quake III Arena
  // see https://en.wikipedia.org/wiki/Fast_inverse_square_root
  int i;
  float x2, y;
  const float threehalfs = 1.5F;

  x2 = number * 0.5F;
  y = number;
  i = *(int*) &y;                        // evil floating point bit level hacking
  i = 0x5f3759df - (i >> 1);             // what the fuck? 
  y = *(float*) &i;
  y = y * (threehalfs - (x2 * y * y));   // 1st iteration

  return y;
}

__device__
float Q_sqrt(float x) {
  return x * Q_rsqrt(x);
}

__device__
unsigned char saturate_uchar(float x) {
  return (unsigned char) min(max(round(x), 0.0f), 255.0f);
}

__global__
void magnitudeKernel(float* x, float* y, unsigned char* r, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < height && col < width) {
    int idx = row * width + col;
    r[idx] = saturate_uchar(Q_sqrt(x[idx] * x[idx] + y[idx] * y[idx]));
  }
}

void sobel(unsigned char *h_img, unsigned char *h_img_sobel, int width, int height, bool measure) {
  unsigned char *d_img, *d_img_sobel;
  float *d_img_sobel_x, *d_img_sobel_y;
  float *d_sobel_x, *d_sobel_y;
  long long size = width * height;
  cudaError_t err;
  cudaEvent_t start, stop;

  err = cudaMalloc((void**) &d_img, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel_x, size * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel_y, size * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_sobel_x, 9 * sizeof(float)); checkError(err);
  err = cudaMalloc((void**) &d_sobel_y, 9 * sizeof(float)); checkError(err);
  
  err = cudaMemcpy(d_img, h_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice); checkError(err);

  float h_sobel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  err = cudaMemcpy(d_sobel_x, h_sobel_x, 9 * sizeof(float), cudaMemcpyHostToDevice); checkError(err);
  err = cudaMemcpy(d_sobel_y, h_sobel_y, 9 * sizeof(float), cudaMemcpyHostToDevice); checkError(err);

  int block_size = 32;
  dim3 dim_grid(ceil((double) width / block_size), ceil((double) height / block_size), 1);
  dim3 dim_block(block_size, block_size, 1);

  if (measure) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
  }

  convolutionKernel<<<dim_grid, dim_block>>>(d_img, d_sobel_x, d_img_sobel_x, 3, width, height);
  cudaDeviceSynchronize();
  
  convolutionKernel<<<dim_grid, dim_block>>>(d_img, d_sobel_y, d_img_sobel_y, 3, width, height);
  cudaDeviceSynchronize();
  
  magnitudeKernel<<<dim_grid, dim_block>>>(d_img_sobel_x, d_img_sobel_y, d_img_sobel, width, height);
  cudaDeviceSynchronize();
  
  if (measure) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float seconds = 0;
    cudaEventElapsedTime(&seconds, start, stop);
    seconds *= 1E-3;
    int num_arrays_float = 4;
    int num_arrays_uchar = 2;
    int bytes = num_arrays_float * sizeof(float) + num_arrays_uchar * sizeof(unsigned char);
    int num_ops = 2 * 9 * 2 + 6;
    
    float bw = size * bytes / seconds * 1E-9;
    float th = num_ops * size / seconds * 1E-9;
    printf("Effective bandwidth & Computational throughput\n");
    printf("%2.5f (GB/s)     & %2.5f (GFLOPS/s)\n", bw, th);
  }

  err = cudaMemcpy(h_img_sobel, d_img_sobel, size * sizeof(unsigned char), cudaMemcpyDeviceToHost); checkError(err);
  
  err = cudaFree(d_img); checkError(err);
  err = cudaFree(d_img_sobel_x); checkError(err);
  err = cudaFree(d_img_sobel_y); checkError(err);
  err = cudaFree(d_img_sobel); checkError(err);
}

void runProgram(Mat& image, bool show, bool measure) {
  int height = image.rows;
  int width = image.cols;
 
  unsigned char *img_sobel = (unsigned char*) malloc(width * height * sizeof(unsigned char));
  unsigned char *img = (unsigned char*) image.data;

  sobel(img, img_sobel, width, height, measure);

  if (show) {
    imshow("Input", Mat(height, width, CV_8UC1, img));
    waitKey(0);
    imshow("Sobel operator", Mat(height, width, CV_8UC1, img_sobel));
    waitKey(0);
  }

  free(img_sobel);
}

void usage(char* program_name) {
  int n = 2;
  string opts[] = {"-s, --show", "-m, --measure"};
  string description[] = {
    "Show original image and result",
    "Permormance measures"
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
      {"measure", no_argument, 0, 'm'},
      {0, 0, 0, 0}
  };
  
  bool show = false;
  bool measure = false;
  while ((opt = getopt_long(argc, argv, "sm", options, &opt_index)) != -1) {
    switch (opt) {
      case 's':
        show = true;
        break;
      case 'm':
        measure = true;
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

  Mat image = imread(argv[optind], CV_LOAD_IMAGE_GRAYSCALE);
  if (!image.data) {
    printf("Could not open or find %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  
  runProgram(image, show, measure);
  
  return 0;
}

