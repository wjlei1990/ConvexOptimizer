cmake_minimum_required(VERSION 3.8)
project(test_dot)

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
        src/cxx/main.cpp
        src/cxx/direction/direcUpdate.cpp
        src/cxx/function/functionwrapper.cpp
        src/cxx/function/optimizer.cpp)

add_executable(ConvexOptimizer ${SOURCE_FILES})
target_link_libraries(ConvexOptimizer
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES})

set(SOURCE_FILES_2
        src/cxx/testing/test_dot.cpp)
#        src/cxx/direction/direcUpdate.cpp)

add_executable(test_dot ${SOURCE_FILES_2})
target_link_libraries(test_dot
        ${MPI_CXX_LIBRARIES}
        ${Boost_LIBRARIES})
