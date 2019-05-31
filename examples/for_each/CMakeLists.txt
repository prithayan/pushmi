

add_executable(for_each_2 for_each_2.cpp)
target_link_libraries(for_each_2
  pushmi
  examples
  Threads::Threads)

add_executable(for_each_3 for_each_3.cpp)
target_link_libraries(for_each_3
  pushmi
  examples
  Threads::Threads)

add_executable(for_each_4 for_each_4.cpp)
target_link_libraries(for_each_4
  pushmi
  examples
  Threads::Threads)

add_executable(for_each_5 for_each_5.cpp)
target_link_libraries(for_each_5
  pushmi
  examples
  Threads::Threads)
