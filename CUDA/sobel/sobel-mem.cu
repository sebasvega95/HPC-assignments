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
#define BLOCK_SIZE 32
#define KERNEL_ORDER 3
#define KERNEL_SIDE 1

using namespace cv;
using namespace std;

__device__
bool inside_image(int row, int col, int width, int height) {
  return row >= 0 && row < height && col >= 0 && col < width;
}

__device__
unsigned char bound_to_image(unsigned char* image, int row, int col, int width, int height) {
  if (inside_image(row, col, width, height))
    return image[row * width + col];
  else
    return 0;
}

__device__
float Q_rsqrt(float number) {
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

__constant__ float sobel_x[KERNEL_ORDER * KERNEL_ORDER];
__constant__ float sobel_y[KERNEL_ORDER * KERNEL_ORDER];

__global__
void sobelOperatorKernel(unsigned char* image, unsigned char* out_image, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float tile[BLOCK_SIZE + 2 * KERNEL_SIDE][BLOCK_SIZE + 2 * KERNEL_SIDE];
  
  if (row < height && col < width) {
    // Loading tile
    int y_centre = threadIdx.y + 1;
    int x_centre = threadIdx.x + 1;
    tile[y_centre][x_centre] = image[row * width + col];

    // Check corners
    if (threadIdx.y == 0 && threadIdx.x == 0) {
      tile[y_centre - 1][x_centre - 1] = bound_to_image(image, row - 1, col - 1, width, height);
    } else if (threadIdx.y == 0 && threadIdx.x == blockDim.x - 1) {
      tile[y_centre - 1][x_centre + 1] = bound_to_image(image, row - 1, col + 1, width, height);
    } else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == 0) {
      tile[y_centre + 1][x_centre - 1] = bound_to_image(image, row + 1, col - 1, width, height);
    } else if (threadIdx.y == blockDim.y - 1 && threadIdx.x == blockDim.x - 1) {
      tile[y_centre + 1][x_centre + 1] = bound_to_image(image, row + 1, col + 1, width, height);
    }
    
    // Check sides
    if (threadIdx.y == 0) {
      tile[y_centre - 1][x_centre] = bound_to_image(image, row - 1, col, width, height);
    } else if (threadIdx.y == blockDim.y - 1) {
      tile[y_centre + 1][x_centre] = bound_to_image(image, row + 1, col, width, height);
    }
    if (threadIdx.x == 0) {
      tile[y_centre][x_centre - 1] = bound_to_image(image, row, col - 1, width, height);
    } else if (threadIdx.x == blockDim.x - 1) {
      tile[y_centre][x_centre + 1] = bound_to_image(image, row, col + 1, width, height);
    }
    
    __syncthreads();
    
    // Calculate gradient in x-direction
    float grad_x = 0;
    for (int i = -KERNEL_SIDE; i <= KERNEL_SIDE; i++) {
      for (int j = -KERNEL_SIDE; j <= KERNEL_SIDE; j++) {
        grad_x += tile[y_centre + i][x_centre + j] * sobel_x[(KERNEL_SIDE + i) * KERNEL_ORDER + (KERNEL_SIDE + j)];
      }
    }
    
    // Calculate gradient in y-direction
    float grad_y = 0;
      for (int i = -KERNEL_SIDE; i <= KERNEL_SIDE; i++) {
      for (int j = -KERNEL_SIDE; j <= KERNEL_SIDE; j++) {
        grad_y += tile[y_centre + i][x_centre + j] * sobel_y[(KERNEL_SIDE + i) * KERNEL_ORDER + (KERNEL_SIDE + j)];
      }
    }
    
    // Calculate gradient magnitude
    out_image[row * width + col] = (unsigned char) Q_sqrt(grad_x * grad_x + grad_y * grad_y);
  }
}

void sobel(unsigned char *h_img, unsigned char *h_img_sobel, int width, int height, bool measure) {
  unsigned char *d_img, *d_img_sobel;
  long long size = width * height;
  cudaError_t err;
  cudaEvent_t start, stop;

  err = cudaMalloc((void**) &d_img, size * sizeof(unsigned char)); checkError(err);
  err = cudaMalloc((void**) &d_img_sobel, size * sizeof(unsigned char)); checkError(err);
  
  err = cudaMemcpy(d_img, h_img, size * sizeof(unsigned char), cudaMemcpyHostToDevice); checkError(err);

  float h_sobel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float h_sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  err = cudaMemcpyToSymbol(sobel_x, h_sobel_x, 9 * sizeof(float)); checkError(err);
  err = cudaMemcpyToSymbol(sobel_y, h_sobel_y, 9 * sizeof(float)); checkError(err);

  dim3 dim_grid(ceil((double) width / BLOCK_SIZE), ceil((double) height / BLOCK_SIZE), 1);
  dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

  if (measure) {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
  }

  sobelOperatorKernel<<<dim_grid, dim_block>>>(d_img, d_img_sobel, width, height);
  cudaDeviceSynchronize();
  
   if (measure) {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float seconds = 0;
    cudaEventElapsedTime(&seconds, start, stop);
    seconds *= 1E-3;
    
    int num_arrays = 2;
    int bytes = num_arrays * sizeof(unsigned char);
    int num_ops = 9 * 2;
    
    float bw = size * bytes / seconds * 1E-9;
    float th = size * num_ops / seconds * 1E-9;
    printf("Effective bandwidth & Computational throughput\n");
    printf("%2.5f (GB/s)     & %2.5f (GFLOPS/s)\n", bw, th);
  }

  err = cudaMemcpy(h_img_sobel, d_img_sobel, size * sizeof(unsigned char), cudaMemcpyDeviceToHost); checkError(err);
  
  err = cudaFree(d_img); checkError(err);
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
  int n = 1;
  string opts[] = {"-s, --show", "-s, --measure"};
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

