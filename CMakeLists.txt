cmake_minimum_required(VERSION 3.8)
project(ConvexOptimizer)

set(CMAKE_CXX_STANDARD 11)

find_package(Boost REQUIRED COMPONENTS mpi serialization system)
find_package(MPI REQUIRED)

include_directories(${BOOST_INCLUDE_DIR}
        ${MPI_INCLUDE_PATH})

include_directories(src/cxx/direction)
include_directories(src/cxx/function)
include_directories(src/cxx/utilities)

# make
set(SOURCE_FILES
        src/cxx/main.cpp)

add_executable(ConvexOptimizer ${SOURCE_FILES})
target_link_libraries(ConvexOptimizer
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES})

# ###########################################
# test
set(gtest_lib_dir /Users/wenjie/software/googletest/googletest/build)
set(gtest_inc_dir /Users/wenjie/software/googletest/googletest/include)
include_directories(${gtest_inc_dir})

set(SOURCE_FILES_2
        src/cxx/testing/test_dot.cpp src/cxx/testing/test_optimizer.cpp)
add_executable(test_dot ${SOURCE_FILES_2})
target_link_libraries(test_dot
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES})
target_link_libraries(test_dot ${gtest_lib_dir}/libgtest.a ${gtest_lib_dir}/libgtest_main.a)

set(SOURCE_FILES_3
        src/cxx/testing/test_mpi_direcUpdate.cpp)
add_executable(test_mpi ${SOURCE_FILES_3})
target_link_libraries(test_mpi
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES})
target_link_libraries(test_mpi ${gtest_lib_dir}/libgtest.a ${gtest_lib_dir}/libgtest_main.a)
