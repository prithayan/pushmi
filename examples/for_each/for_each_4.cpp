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
unsigned maxNumExps = 2;
#define index2Row(index) ((index)/MAXDIM )
#define index2Col(index) ((index)%MAXDIM )

int main(int argc, char **argv) {
  if (argc > 1) {
    MAXDIM = strtol(argv[1], nullptr, 0);
    std::cout<<"\n Using Matrix Dimensions:"<<MAXDIM;
  }
  //mi::pool p{std::max(1u, 1u)};//std::thread::hardware_concurrency())};

    bool mismatch = false;
  std::cout<<"\n num of threads:"<<std::thread::hardware_concurrency();
  unsigned numRows = MAXDIM, numCols = MAXDIM;
  std::vector<float> C1(numRows*numCols, 0);
  std::vector<float> C2(numRows*numCols, 0);
  std::vector<std::vector<float>> C2d(numRows);
  std::vector<std::vector<float>> C2d2(numRows);
  std::vector<float> referenceC(numRows*numCols, 0);
  std::vector<float> A(numRows*numCols);
  std::vector<float> B(numRows*numCols);

  double executor1TimeSum = 0, executor2TimeSum = 0,executor3TimeSum = 0,executor4TimeSum = 0, baseTimeSum =0 ;
  for (unsigned numExperiments = 0 ; numExperiments < maxNumExps ; numExperiments++) {

    mi::pool p{std::max(1u, std::thread::hardware_concurrency())};
    auto f = []()-> float{return rand() % 10000; };
    generate(A.begin(), A.end(), f);
    generate(B.begin(), B.end(), f);
    std::fill(referenceC.begin(), referenceC.end(), 0);
    {
      //std::array<std::array<float, numCols>, numRows> C;
        for (unsigned i =0 ; i < numCols; i++) {
          C2d[i].resize(numCols);
          for (unsigned j =0 ; j < numCols; j++)
            C2d[i][j] = 0 ;
      }

      //for (auto CCols: referenceC)
      //  for (unsigned i =0 ; i < numCols; i++)
      //    CCols[i] = 0 ;
      auto start = high_resolution_clock::now(); 

      mi::for_each(
          inline_bulk_target(), C2d.begin(),
          C2d.end(),
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
      executor3TimeSum += microTime;
      std::cout<<std::fixed;
      std::cout << "\n Time taken by Executor3 function: "
        << microTime << " seconds \n"; 
    }
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
      std::cout << "\n Time taken by Executor4 function: "
        << microTime << " seconds \n"; 
    }
    {
      std::fill(C1.begin(), C1.end(), 0);
      auto start = high_resolution_clock::now(); 

      mi::for_each(
          naive_executor_bulk_target(p.executor()),
          C1.begin(),
          C1.end(),
          [&A, &B, numRows, numCols](float& x, unsigned index) 
          {
          for (unsigned k = 0 ; k < numRows; k++ ){
            //C[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
            unsigned row = index2Row(index);
            unsigned col = index2Col(index);
            x+= A[row*numCols + k]*B[k*numCols+col];
            //std::cout<<" "<<row<<","<<col<<", A="<<A[row*numCols + k]
            //  <<", B="<<B[k*numCols+col]<<", x="<<x;
          }
          //std::cout<<"\n";
          //std::cout<<"\t C["<<index2Row(index)<<"]["<<index2Col(index)<<"]="<<x ; 
          });
      auto stop = high_resolution_clock::now(); 
      auto duration = duration_cast<microseconds>(stop - start); 
      double microTime =(double) duration.count()/1000000;
      executor1TimeSum += microTime;
      std::cout<<std::fixed;
      std::cout << "\n Time taken by Executor1 function: "
        << microTime << " seconds \n"; 
    }
    {
      std::fill(C2.begin(), C2.end(), 0);
      auto start = high_resolution_clock::now(); 

      mi::for_each(
          inline_bulk_target(), C2.begin(),
          C2.end(),
          [&A, &B, numRows, numCols](float& x, unsigned index) 
          {
          for (unsigned k = 0 ; k < numRows; k++ ){
          //C[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
          unsigned row = index2Row(index);
          unsigned col = index2Col(index);
          x+= A[row*numCols + k]*B[k*numCols+col];
          //std::cout<<" "<<row<<","<<col;//<<", A="<<A[row*numCols + k]
          //  <<", B="<<B[k*numCols+col]<<", x="<<x;
          }
          //std::cout<<"\n";
          //std::cout<<"\t C["<<index2Row(index)<<"]["<<index2Col(index)<<"]="<<x ; 
          });
      auto stop = high_resolution_clock::now(); 
      auto duration = duration_cast<microseconds>(stop - start); 
      double microTime =(double) duration.count()/1000000;
      executor2TimeSum += microTime;
      std::cout<<std::fixed;
      std::cout << "\n Time taken by Executor2 function: "
        << microTime << " seconds \n"; 
    }

    {
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
    for (unsigned i = 0 ; i < numRows; i++ ){
      mismatch = false;
      for (unsigned j = 0 ; j < numRows; j++ ){
        if (fabs(referenceC[i*numCols+j]- C1[i*numCols+j]) > 0.001 ||
          fabs(referenceC[i*numCols+j]- C2[i*numCols+j]) > 0.001 || 
          fabs(referenceC[i*numCols+j]- C2d[i][j]) > 0.001 ||
          fabs(referenceC[i*numCols+j]- C2d2[i][j]) > 0.001
          ){
          std::cout<<"\n Difference :"<<i<<","<<j<<"="<<referenceC[i*numCols+j]<<" but C1="
            << C1[i*numCols+j];
          mismatch = true; break;
        }
      }
      if (mismatch) break;
    }
  }
    auto E1microTime =(double) executor1TimeSum / maxNumExps;
    auto E2microTime =(double) executor2TimeSum / maxNumExps;
    auto BmicroTime =(double) baseTimeSum / maxNumExps;
    auto E3microTime =(double) executor3TimeSum / maxNumExps;
    auto E4microTime =(double) executor4TimeSum / maxNumExps;
    std::ofstream csvFile; 
    csvFile.open("executorRuntime.csv", std::ios::out|std::ios::app);
    //csvFile << "Experiment, Matrix Dimensions, Implementation1, Implementation2, Baseline ";
    csvFile<<std::fixed;
    csvFile<<"\npushmi,"<<MAXDIM<<","<< E1microTime<<","<< E2microTime<<","<< E3microTime<<","<<E4microTime<<","<<BmicroTime;
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
