cmake_minimum_required(VERSION 3.12)
project(CUDA_Nearest_Neighbors)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

find_package (OpenMP)
find_package(CUDA)

if(CUDA_FOUND)
    # using CUDA

    if(OPENMP_FOUND)
        # using OpenMP
        set(CMAKE_CUDA_FLAGS "-Xcompiler -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "-fopenmp")
    endif(OPENMP_FOUND)

    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_61")
    enable_language(CUDA)
    add_executable(CUDA_Nearest_Neighbors src/main.cu src/search.hpp)

else()
    # using CPU

    if(OPENMP_FOUND)
        # using OpenMP
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif(OPENMP_FOUND)

    add_executable(CUDA_Nearest_Neighbors src/main.cpp src/search.hpp)

endif(CUDA_FOUND)
