cmake_minimum_required(VERSION 3.12)
project(CUDA_Nearest_Neighbors)

set(CMAKE_CXX_STANDARD 17)

include_directories(src)

find_package(CUDA)
if(CUDA_FOUND)
    # using CUDA
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    enable_language(CUDA)
    add_executable(CUDA_Nearest_Neighbors src/main.cu)
else()
    # using CPU
    add_executable(CUDA_Nearest_Neighbors src/main.cpp)
endif(CUDA_FOUND)
