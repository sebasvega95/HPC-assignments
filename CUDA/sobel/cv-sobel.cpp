#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <iostream>
#include <getopt.h>
#include <cstdio>
#include <cmath>

using namespace cv;
using namespace std;

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

  Mat h_image = imread(argv[optind], CV_LOAD_IMAGE_GRAYSCALE);
  if (!h_image.data) {
    printf("Could not open or find %s\n", argv[optind]);
    exit(EXIT_FAILURE);
  }
  
  gpu::GpuMat d_image, d_img_sobel_x, d_img_sobel_y, d_img_sobel;
  d_image.upload(h_image);

  gpu::Sobel(d_image, d_img_sobel_x, CV_32F, 1, 0);
  gpu::Sobel(d_image, d_img_sobel_y, CV_32F, 0, 1);

  gpu::magnitude(d_img_sobel_x, d_img_sobel_y, d_img_sobel);
  
  if (show) {
    Mat h_img_sobel(d_img_sobel);
    convertScaleAbs(h_img_sobel, h_img_sobel);
    imshow("Input", Mat(h_image));
    waitKey(0);
    imshow("Sobel operator", h_img_sobel);
    waitKey(0);
  }
  
  return 0;
}

