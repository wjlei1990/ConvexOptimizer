cmake_minimum_required(VERSION 3.8)
project(ConvexOptimizer)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        src/functionwrapper.cpp
        src/functionwrapper.h
        src/main.cpp
        src/optimizer.cpp
        src/optimizer.h
        src/utils.cpp
        src/utils.h)

add_executable(ConvexOptimizer ${SOURCE_FILES})