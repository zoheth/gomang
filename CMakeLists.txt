cmake_minimum_required(VERSION 3.18)
project(gomang)

set(CMAKE_CXX_STANDARD 20)

set(THIRD_PARTY_PATH "${CMAKE_SOURCE_DIR}/third_party")
if(NOT EXISTS ${THIRD_PARTY_PATH})
    file(MAKE_DIRECTORY ${THIRD_PARTY_PATH})
endif()

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++20 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++20 -Wno-deprecated")
endif ()

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 89) # For RTX 40xx series
enable_language(CUDA)

option(ENABLE_TENSORRT "enable TensorRT engine" OFF)
option(ENABLE_MNN "enable MNN engine" OFF)
option(ENABLE_NCNN "enable NCNN engine" OFF)
option(BUILD_EXAMPLES "Build examples" ON)

add_subdirectory(gomang)

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()




