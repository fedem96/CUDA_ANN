cmake_minimum_required(VERSION 3.12)
project(CUDA_Nearest_Neighbors)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)
#include_directories(nanoflann/include)
#include_directories(nanoflann/examples)
include_directories(~/Programmi/eigen/Eigen)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

add_executable(CUDA_nanoflann src/main.cu)