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

#include "../for_each.h"
#include "../pool.h"

using namespace pushmi::aliases;
using namespace std::chrono; 

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

#define MAXDIM 1024
#define index2Row(index) ((index)/MAXDIM )
#define index2Col(index) ((index)%MAXDIM )

int main() {
  mi::pool p{std::max(1u, std::thread::hardware_concurrency())};
  //mi::pool p{std::max(1u, 1u)};//std::thread::hardware_concurrency())};

  std::cout<<"\n num of threads:"<<std::thread::hardware_concurrency();
  const unsigned numRows = MAXDIM, numCols = MAXDIM;
  std::vector<float> C(numRows*numCols, 0);
  std::vector<float> referenceC(numRows*numCols, 0);
  std::vector<float> A(numRows*numCols);
  std::vector<float> B(numRows*numCols);

  auto f = []()-> float{return rand() % 10000; };
  generate(A.begin(), A.end(), f);
  generate(B.begin(), B.end(), f);
  //for (unsigned i = 0 ; i < numRows*numCols; i++ ){
  //  C[i] = 0;
  //  referenceC[i] = 0;
  //  A[i] =1;// index2Row(i)+i;
  //  B[i] =i;// index2Col(i)*i;
  //}
    auto start = high_resolution_clock::now(); 

  mi::for_each(
      naive_executor_bulk_target(p.executor()),
      C.begin(),
      C.end(),
      [&A, &B](float& x, unsigned index) 
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
   std::cout << "Time taken by function: "
                << (duration.count())/1000000 << " microseconds \n"; 

    auto baseStart = high_resolution_clock::now(); 
  for (unsigned i = 0 ; i < numRows; i++ ){
    for (unsigned j = 0 ; j < numRows; j++ ){
      for (unsigned k = 0 ; k < numRows; k++ ){
        referenceC[i*numCols+j] +=  A[i*numCols+k]* B[k*numCols+j];
      }
     // if (referenceC[i*numCols+j] != C[i*numCols+j]){
     //   std::cout<<"\n Difference :"<<i<<","<<j<<"="<<referenceC[i*numCols+j];
     //   goto outofLoop1;
     // }
    }
  }
   auto baseStop = high_resolution_clock::now(); 
   auto baseDuration = duration_cast<microseconds>(baseStop - baseStart); 
   std::cout << "Time taken by function: "
                << baseDuration.count()/1000000 << " microseconds \n"; 
  outofLoop1:


  p.stop();
  p.wait();
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
  std::cout << "\n OK" << std::endl;
}
