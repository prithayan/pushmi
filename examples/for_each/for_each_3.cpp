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

#include "../for_each.h"

using namespace pushmi::aliases;

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

int main() {
  std::vector<int> vec(10);
  std::vector<int> input(10);
  for (unsigned i = 0 ; i < 10 ; i++)
    input[i] = 42;

  mi::for_each(
      inline_bulk_target(), vec.begin(), vec.end(), [=](int& x, unsigned index) 
      { std::cout<< " \n index:"<<input[index]; x = input[index]; });

  assert(
      std::count(vec.begin(), vec.end(), 42) == static_cast<int>(vec.size()));

  std::cout << "OK" << std::endl;
}
