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
unsigned MAXDIM = 1024;
unsigned maxNumExps = 1;
#define index2Row(index) ((index)/MAXDIM )
#define index2Col(index) ((index)%MAXDIM )

void sgemm_omp_compute_tiled(const std::vector<float> &matrix_A, const std::vector<float> &matrix_B, 
                          std::vector<float> &matrix_C, const int numRows, const int numCols )
{

  //using namespace std::chrono;
  //high_resolution_clock::time_point t1 = high_resolution_clock::now();

  ///float alpha = config.alpha;
  //float beta = config.beta;
  int size1 = numRows;
  int size2 = numCols;
  int size3 = numCols;

  //float * matrix_A = config.matrix_A;
  //float * matrix_B = config.matrix_B;
  //float * matrix_C = config.matrix_C;

#define  TILE_SIZE_A  256 //128
#define  TILE_SIZE_B  16 //16
  for (int iter = 0; iter < 2; ++iter)
  {   
#pragma omp parallel for schedule(static) collapse(2) //shared(matrix_A, matrix_B, matrix_C)
    for (int ii = 0; ii < size1; ii += TILE_SIZE_A)
      for (int jj = 0; jj < size3; jj += TILE_SIZE_B)
      {   

        std::vector<float> part_B(TILE_SIZE_B * size2, 0);
        for(int bj = 0; bj < size2; bj ++)
        {
          for(int bi = 0; bi < TILE_SIZE_B; bi ++)
          {
            part_B[bi * size2 + bj] = matrix_B[bj * size3 + bi + jj];
          }
        }
        for (int i = ii; i < ii + TILE_SIZE_A; i++)
        {
          for (int j = jj; j < jj + TILE_SIZE_B; j++)
          {
            float sum = 0;
#pragma omp simd simdlen(32) 
            for (int k = 0; k < size2; k++)
            {
              //sum += (part_A[(i - ii) * size2 + k] * part_B[(j-jj) * size2+ k]);
              sum += (matrix_A[i * size2 + k] * part_B[(j-jj) * size2 + k]);
              //sum += (matrix_A[i * size2 + k] * matrix_B[k * size3 + j]);
            }   
            matrix_C[i*size3 + j] = sum;
          }   
        }   
      }   
  }

}

void matrix_mult_parallel2(const std::vector<float> &A, const std::vector<float> &B, 
                          std::vector<float> &C, const int numRows, const int numCols  )
{
  //Dynamic Scheduler
  int i,j,k;
  #pragma omp parallel for schedule(dynamic,50) private(i,j,k) shared(A,B,C)
  for(i=0;i<numRows;i++)
    for( j=0;j<numRows;j++)
      for(k=0;k<numRows;k++)
        C[i*numCols+j]+=A[i*numCols+k]*B[k*numCols+j];
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
    csvFile<<"\npushmi,"<<MAXDIM<<",";//<<MAXTHREADS<<","<<E4microTime;//<<","<<BmicroTime;
    bool mismatch = false;
  unsigned numRows = MAXDIM, numCols = MAXDIM;
  std::vector<std::vector<float>> C2d2(numRows);
  std::vector<float> referenceC(numRows*numCols, 0);
  std::vector<float> A(numRows*numCols);
  std::vector<float> B(numRows*numCols);

  for (unsigned MAXTHREADS = 12 ; MAXTHREADS <= std::thread::hardware_concurrency(); MAXTHREADS++){
  double ompTimeSum = 0;
  double executor1TimeSum = 0, executor2TimeSum = 0,executor3TimeSum = 0,executor4TimeSum = 0, baseTimeSum =0 ;
  for (unsigned numExperiments = 0 ; numExperiments < maxNumExps ; numExperiments++) {

    mi::pool p{std::max(1u, MAXTHREADS )};
    auto f = []()-> float{return rand() % 10000; };
    generate(A.begin(), A.end(), f);
    generate(B.begin(), B.end(), f);
    std::fill(referenceC.begin(), referenceC.end(), 0);
    {
        for (unsigned i =0 ; i < numCols; i++) {
          C2d2[i].resize(numCols);
          for (unsigned j =0 ; j < numCols; j++)
            C2d2[i][j] = 0 ;
      }
      auto start = high_resolution_clock::now(); 
      if (0)
      mi::for_each(
          naive_executor_bulk_target(p.executor()),
          C2d2.begin(),
          C2d2.end(),
          [&A, &B, numRows, numCols](std::vector<float> &x, unsigned index) 
          {
              //std::cout<<" size of x = "<<x.size()<<"\n" ; 
            for (unsigned j = 0 ; j < numRows; j++ ){
                unsigned col = j;
              float reductionSum = 0;
              for (unsigned k = 0 ; k < numRows; k++ ){
                //C[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
                //unsigned row = index2Row(index);
                //unsigned col = index2Col(index);
                unsigned row = index;
                reductionSum += A[row*numCols + k]*B[k*numCols+col];
                //std::cout<<" "<<row<<","<<col;//<<", A="<<A[row*numCols + k]
                  //<<", B="<<B[k*numCols+col]<<", x="<<reductionSum;
              }
              x[col] = reductionSum;
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
    }
    {
        for (unsigned i =0 ; i < numCols; i++) {
          //C2d2[i].resize(numCols);
          for (unsigned j =0 ; j < numCols; j++)
            C2d2[i][j] = 0 ;
      }
      auto start = high_resolution_clock::now(); 
      if (0)
      mi::for_each(
          naive_executor_bulk_target(p.executor()),
          C2d2.begin(),
          C2d2.end(),
          [&A, &B, numRows, numCols](std::vector<float> &x, unsigned index) 
          {
              //std::cout<<" size of x = "<<x.size()<<"\n" ; 
            for (unsigned j = 0 ; j < numRows; j++ ){
                unsigned col = j;
              float reductionSum = 0;
              for (unsigned k = 0 ; k < numRows; k++ ){
                //C[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
                //unsigned row = index2Row(index);
                //unsigned col = index2Col(index);
                unsigned row = index;
                reductionSum += A[row*numCols + k]*B[k*numCols+col];
                //std::cout<<" "<<row<<","<<col;//<<", A="<<A[row*numCols + k]
                  //<<", B="<<B[k*numCols+col]<<", x="<<reductionSum;
              }
              x[col] = reductionSum;
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
    }

    {
      auto baseStart = high_resolution_clock::now(); 
      if (0)
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
  {
  std::vector<float> C(numRows*numCols, 0);
    auto start = high_resolution_clock::now(); 

    //matrix_mult_parallel2(A, B, C, numRows, numCols); 
    sgemm_omp_compute_tiled(A, B, C, numRows, numCols); 
    
    auto stop = high_resolution_clock::now(); 
    auto duration = duration_cast<microseconds>(stop - start); 
    double microTime =(double) duration.count()/1000000;
    ompTimeSum += microTime;
    std::cout << "\n Time taken by OMP Executor function: "
      << microTime << " seconds \n"; 
  }
  
    auto E4microTime =(double) executor4TimeSum / maxNumExps;
    auto OmpmicroTime =(double) ompTimeSum; // / maxNumExps;
    //std::ofstream csvFile; 
    //csvFile.open("executorRuntime.csv", std::ios::out|std::ios::app);
    //csvFile << "Experiment, Matrix Dimensions, Implementation1, Implementation2, Baseline ";
    //csvFile<<std::fixed;
    csvFile<<MAXTHREADS<<","<<E4microTime<<","<<OmpmicroTime<<","<<baseTimeSum;
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
