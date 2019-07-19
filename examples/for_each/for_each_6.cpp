/*
 * Copyright 2018-present Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <math.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <string.h>
#include "mandelCPP.hpp"

#include "../for_each.h"
//#include <pushmi/o/for_each.h>
#include "../pool.h"

using namespace pushmi::aliases;
using namespace std::chrono; 


auto inline_bulk_target() {
  return [](auto init,
            auto selector,
            auto input,
            auto&& func,
            auto sb,
            auto se,
            auto out) {
    try {
      auto acc = init(input);
      unsigned index = 0 ; 
      for (decltype(sb) idx{sb}; idx != se; ++idx, ++index) {
        func(acc, idx, index);
      }
      auto result = selector(std::move(acc));
      mi::set_value(out, std::move(result));
      mi::set_done(out);
    } catch (...) {
      mi::set_error(out, std::current_exception());
    }
  };
}

  static float scaleRow(int i, int row_count) {     // scale from 0..row_count to -1.5..0.5
    return -1.5f + (i*(2.0f / float(row_count)));
  }
  static float scaleCol(int i, int col_count) {     // scale from 0..col_count to -1..1
    return -1 + (i*(2.0f / float(col_count)));
  }

  static std::complex<float> complex_square( std::complex<float> c)
  {
    return std::complex<float>( c.real()*c.real() - c.imag()*c.imag(), c.real()*c.imag()*2 );
  }

  // A point is in the Mandelbrot set if it does not diverge within max_iterations
  static int point(std::complex<float>c, int max_iteration) {
    int count = 0;
    std::complex<float> z = 0;
    for (; count < max_iteration; count++ ) {
      float r = z.real(); float im = z.imag();
      if (((r*r) + (im*im)) >= 4.0) break;    // leave loop if diverging
      z = complex_square(z) + c;
    }
    return count;
  }
template <class Executor, class Allocator = std::allocator<char>>
auto naive_executor_bulk_target(Executor e, Allocator a = Allocator{}) {
  return [e, a](
             auto init,
             auto selector,
             auto input,
             auto&& func,
             auto sb,
             auto se,
             auto out) mutable {
    using RS = decltype(selector);
    using F = std::conditional_t<
        std::is_lvalue_reference<decltype(func)>::value,
        decltype(func),
        typename std::remove_reference<decltype(func)>::type>;
    using Out = decltype(out);
    try {
      typename std::allocator_traits<Allocator>::template rebind_alloc<char>
          allocState(a);
      auto shared_state = std::allocate_shared<std::tuple<
          std::exception_ptr, // first exception
          Out, // destination
          RS, // selector
          F, // func
          std::atomic<decltype(init(input))>, // accumulation
          std::atomic<std::size_t>, // pending
          std::atomic<std::size_t> // exception count (protects assignment to
                                   // first exception)
          >>(
          allocState,
          std::exception_ptr{},
          std::move(out),
          std::move(selector),
          (decltype(func)&&)func,
          init(std::move(input)),
          1,
          0);
      e.schedule() | op::submit([e, sb, se, shared_state](auto) mutable {
        auto stepDone = [](auto shared_state) {
          // pending
          if (--std::get<5>(*shared_state) == 0) {
            // first exception
            if (std::get<0>(*shared_state)) {
              mi::set_error(
                  std::get<1>(*shared_state), std::get<0>(*shared_state));
              return;
            }
            try {
              // selector(accumulation)
              auto result = std::get<2>(*shared_state)(
                  std::move(std::get<4>(*shared_state).load()));
              mi::set_value(std::get<1>(*shared_state), std::move(result));
              mi::set_done(std::get<1>(*shared_state));
            } catch (...) {
              mi::set_error(
                  std::get<1>(*shared_state), std::current_exception());
            }
          }
        };
         unsigned index=0;
        for (decltype(sb) idx{sb}; idx != se; ++idx, index++) {
          ++std::get<5>(*shared_state);
          e.schedule() | op::submit([shared_state, idx, stepDone, index](auto ex) {
            try {
              // this indicates to me that bulk is not the right abstraction
              auto old = std::get<4>(*shared_state).load();
              auto step = old;
              do {
                step = old;
                // func(accumulation, idx)
                //std::cout<<"\n line 95:";
                std::get<3> (*shared_state)(step, idx, index);
              } while (!std::get<4>(*shared_state)
                            .compare_exchange_strong(old, step));
            } catch (...) {
              // exception count
              if (std::get<6>(*shared_state)++ == 0) {
                // store first exception
                std::get<0>(*shared_state) = std::current_exception();
              } // else eat the exception
            }
            stepDone(shared_state);
          });
        }
        stepDone(shared_state);
      });
    } catch (...) {
      e.schedule() |
          op::submit([out = std::move(out), ep = std::current_exception()](
                         auto) mutable { mi::set_error(out, ep); });
    }
  };
}

//#define MAXDIM 1024
unsigned MAXDIM = 6000;//120;//, 120,1024;
unsigned maxNumExps = 1;
#define index2Row(index) ((index)/MAXDIM )
#define index2Col(index) ((index)%MAXDIM )

  void Print(const std::vector<std::vector<float>> &kb, int row_count, int col_count, int max_iterations) {
    //auto kb = b.template get_access<access::mode::read>();

    for (int i = 0; i < row_count; i++) {
      for (int j = 0; j < col_count; j++) {
        std::cout << std::setw(1) << ((kb[i][j] >= max_iterations) ? "x" : " ");
      }
      std::cout << std::endl;
    }
  }
int main(int argc, char **argv) {
  unsigned MAXTHREADS = std::thread::hardware_concurrency();
  if (argc > 1) {
    MAXDIM = strtol(argv[1], nullptr, 0);
    std::cout<<"\n Using Matrix Dimensions:"<<MAXDIM;
  }
  if (argc > 2) {
    MAXTHREADS = strtol(argv[2], nullptr, 0);
    std::cout<<"\n Using Matrix Dimensions:"<<MAXDIM;
  }
  //mi::pool p{std::max(1u, 1u)};//std::thread::hardware_concurrency())};

    std::ofstream csvFile; 
    csvFile.open("executorRuntime.csv", std::ios::out|std::ios::app);
    //csvFile << "Experiment, Matrix Dimensions, Implementation1, Implementation2, Baseline ";
    csvFile<<std::fixed;
    //csvFile<<"\npushmi,"<<MAXDIM<<",";//<<MAXTHREADS<<","<<E4microTime;//<<","<<BmicroTime;
    bool mismatch = false;
    //MandelSYCL<120, 120, 100> mSYCL; 
    //6000, 6000, 10000
  const unsigned numRows = MAXDIM, numCols = MAXDIM, max_iterations=10000;
  std::vector<std::vector<float>> C2d2(numRows);
  std::vector<float> referenceC(numRows*numCols, 0);
  std::vector<float> C1d1(numRows*numCols, 0);
  std::vector<float> A(numRows*numCols);
  std::vector<float> B(numRows*numCols);


if (0)
  for (int i = 0 ; i < numRows ; i++){
    for (int j = 0 ; j < numCols ; j++) {
        C2d2[i][j] = point(std::complex<float>(scaleRow(i, numRows), scaleCol(j, numCols)), max_iterations);
    }
  }
        MandelCPP<6000, 6000, 10000> mCPP;     
        //mCPP.Print();

      auto start = high_resolution_clock::now(); 
      //mCPP.Evaluate();
      auto stop = high_resolution_clock::now(); 
      auto duration = duration_cast<microseconds>(stop - start); 
      double microTime =(double) duration.count()/1000000;
      std::cout << "\n Time taken by Sequential  is: "
        << microTime << " seconds \n"; 
  double executor1TimeSum = 0, executor2TimeSum = 0,executor3TimeSum = 0,executor4TimeSum = 0, baseTimeSum =0 ;
  if (1)
  for (unsigned MAXTHREADS = 2; MAXTHREADS <= std::thread::hardware_concurrency()+3; MAXTHREADS++){
    for (unsigned numExperiments = 0 ; numExperiments < maxNumExps ; numExperiments++) {

      mi::pool p{std::max(1u, MAXTHREADS )};
      auto f = []()-> float{return rand() % 10000; };
      generate(A.begin(), A.end(), f);
      generate(B.begin(), B.end(), f);
      std::fill(referenceC.begin(), referenceC.end(), 0);
      std::fill(C1d1.begin(), C1d1.end(), 0);
        csvFile << "\n"<< MAXTHREADS <<",";
      {
          for (unsigned i =0 ; i < numCols; i++) {
            C2d2[i].resize(numCols);
            for (unsigned j =0 ; j < numCols; j++)
              C2d2[i][j] = 0 ;
          }
        auto start = high_resolution_clock::now(); 
        mi::for_each(
            naive_executor_bulk_target(p.executor()),
            C2d2.begin(),
            C2d2.end(),
            [numRows, numCols, max_iterations](std::vector<float> &x, unsigned index) 
            {

              unsigned i = index;
#pragma omp simd
              for (int j = 0 ; j < numCols ; j++) {
                x[j] = point(std::complex<float>(scaleRow(i, numRows), scaleCol(j, numCols)), max_iterations);
              }
            //std::cout<<"\n";
            //std::cout<<"\t C["<<index2Row(index)<<"]["<<index2Col(index)<<"]="<<x ; 
            });
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
        double microTime =(double) duration.count()/1000000;
        executor4TimeSum += microTime;
        std::cout<<std::fixed;
        std::cout << "\n Time taken by Executor function with "<<MAXTHREADS <<" is: "
          << microTime << " seconds \n"; 
        csvFile << microTime <<",";
      }
      {
        auto start = high_resolution_clock::now(); 
        mi::for_each(
            naive_executor_bulk_target(p.executor()),
            C1d1.begin(),
            C1d1.end(),
            [numRows, numCols, max_iterations](float &x, unsigned index) 
            {
              unsigned i = index2Row(index);
              unsigned j = index2Col(index);

                x = point(std::complex<float>(scaleRow(i, numRows), scaleCol(j, numCols)), max_iterations);
            //std::cout<<"\n";
            //std::cout<<"\t C["<<index2Row(index)<<"]["<<index2Col(index)<<"]="<<x ; 
            });
        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
        double microTime =(double) duration.count()/1000000;
        executor4TimeSum += microTime;
        std::cout<<std::fixed;
        std::cout << "\n Time taken by Executor function with "<<MAXTHREADS <<" is: "
          << microTime << " seconds \n"; 
        csvFile << microTime <<",";
      }
          //Print(C2d2, numRows, numCols, max_iterations);

      if (0){
        auto baseStart = high_resolution_clock::now(); 
        for (unsigned i = 0 ; i < numRows; i++ ){
          for (unsigned j = 0 ; j < numRows; j++ ){
            for (unsigned k = 0 ; k < numRows; k++ ){
              referenceC[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
            }
          }
        }
        auto baseStop = high_resolution_clock::now(); 
        auto baseDuration = duration_cast<microseconds>(baseStop - baseStart); 
        auto microTime =(double) baseDuration.count()/1000000;
        std::cout << "Time taken by Baseline function: "
          << microTime<< " seconds \n"; 
        baseTimeSum += microTime;
      }

      p.stop();
      p.wait();
      if (0)
      for (unsigned i = 0 ; i < numRows; i++ ){
        mismatch = false;
        for (unsigned j = 0 ; j < numRows; j++ ){
          if (
            fabs(referenceC[i*numCols+j]- C2d2[i][j]) > 0.001
            ){
            std::cout<<"\n Difference :"<<i<<","<<j<<"="<<referenceC[i*numCols+j]<<" but C="
              << C2d2[i][j];
            mismatch = true; break;
          }
        }
        if (mismatch) break;
      }
    }
  
    auto E4microTime =(double) executor4TimeSum / maxNumExps;
    //std::ofstream csvFile; 
    //csvFile.open("executorRuntime.csv", std::ios::out|std::ios::app);
    //csvFile << "Experiment, Matrix Dimensions, Implementation1, Implementation2, Baseline ";
    //csvFile<<std::fixed;
    //csvFile<<MAXTHREADS<<","<<E4microTime<<",";//<<BmicroTime;
  }
    csvFile.close();

  //for (unsigned i = 0 ; i < numRows; i++ ){
  //  for (unsigned j = 0 ; j < numRows; j++ ){
  //    for (unsigned k = 0 ; k < numRows; k++ ){
  //      referenceC[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
  //    }
  //    if (referenceC[i*numCols+j] != C[i*numCols+j]){
  //      std::cout<<"\n Again Difference :"<<i<<","<<j<<"="<<referenceC[i*numCols+j];
  //      goto outofLoop2;
  //    }
  //  }
  //}
  outofLoop2:
  std::cout<<"\n=================\n";
  //for (unsigned i = 0 ; i < numRows; i++ ){
  //  for (unsigned j = 0 ; j < numCols; j++ ){
  //    std::cout<<"  "<<C[i*numCols+j];
  //  }
  //  std::cout<<"\n";
  //}
  if (!mismatch)
    std::cout << "\n OK" << std::endl;
  else 
    std::cout << "\n ERROR " << std::endl;

}
