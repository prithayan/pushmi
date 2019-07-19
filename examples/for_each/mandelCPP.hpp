#pragma once

#include <iostream>
#include <iomanip>
#include <complex>

template <int row_count, int col_count, int max_iterations>
class MandelCPP {
    public:
  int *p;

  static float scaleRow(int i) {     // scale from 0..row_count to -1.5..0.5
    return -1.5f + (i*(2.0f / float(row_count)));
  }
  static float scaleCol(int i) {     // scale from 0..col_count to -1..1
    return -1 + (i*(2.0f / float(col_count)));
  }

  // A point is in the Mandelbrot set if it does not diverge within max_iterations
  static int point(std::complex<float> c, int max_iteration) {
    int count = 0;
    std::complex<float> z = 0;
    for (int i = 0; i < max_iteration; i++) {
      float r = z.real(); float im = z.imag();
      if (((r*r) + (im*im)) >= 4.0) break;    // leave loop if diverging
      z = z*z + c; count++;
    }
    return count;
  }

public:
  MandelCPP() {
       p = new int[row_count*col_count];
  };

   ~MandelCPP(){
       delete [] p;
   }


  // iterate over image and compute mandel for each point
  void Evaluate() {
    for (int i = 0; i < row_count; i++) {
      for (int j = 0; j < col_count; j++) {
        p[i*col_count + j] = point(std::complex<float>(scaleRow(i), scaleCol(j)), max_iterations);
      }
    }
  }

  //use only for debugging with small dimensions
  void Print() {
    for (int i = 0; i < row_count; i++) {
      for (int j = 0; j < col_count; j++) {
        std::cout << std::setw(1) << ((p[i * col_count + j] >= max_iterations) ? "x" : " ");
      }
      std::cout << std::endl;
    }
  }

};


