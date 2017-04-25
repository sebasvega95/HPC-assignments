#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace std;

bool inside_image(int row, int col, int width, int height) {
  return row >= 0 && row < height && col >= 0 && col < width;
}

void convolution(unsigned char* image, float* kernel, float* out_image, int kernel_n, int width, int height) {
  int n = kernel_n / 2;
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
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
}

void image_magnitude(float* x, float* y, unsigned char* r, int width, int height) {
  for (int row = 0; row < height; row++) {
    for (int col = 0; col < width; col++) {
      int idx = row * width + col;
      r[idx] = saturate_cast<unsigned char>(hypot(x[idx], y[idx]));
    }
  }
}

void runProgram(Mat& image, bool show) {
  int height = image.rows;
  int width = image.cols;
  
  float sobel_x[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
  float sobel_y[] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
  
  float *img_sobel_x = (float*) malloc(width * height * sizeof(float));
  float *img_sobel_y = (float*) malloc(width * height * sizeof(float));
  unsigned char *img_sobel = (unsigned char*) malloc(width * height * sizeof(unsigned char));
  unsigned char *img = (unsigned char*) image.data;

  convolution(img, sobel_x, img_sobel_x, 3, width, height);
  convolution(img, sobel_y, img_sobel_y, 3, width, height);
  image_magnitude(img_sobel_x, img_sobel_y, img_sobel, width, height);

  if (show) {
    imshow("Input", Mat(height, width, CV_8UC1, img));
    waitKey(0);
    imshow("Sobel operator", Mat(height, width, CV_8UC1, img_sobel));
    waitKey(0);
  }

  free(img_sobel_x);
  free(img_sobel_y);
  free(img_sobel);
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

  Mat image = imread(argv[optind], CV_LOAD_IMAGE_GRAYSCALE);
  if (!image.data) {
    printf("Could not open or find %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  
  runProgram(image, show);
  
  return 0;
}

