set(CUDA_DIR "" CACHE PATH "If build tensorrt backend, need to define path of cuda library.")
set(TensorRT_DIR "" CACHE PATH "If build tensorrt backend, need to define path of tensorrt library.")

if(NOT CUDA_DIR)
    set(CUDA_DIR "/usr/local/cuda")
    message(STATUS "CUDA_DIR is not defined, use default dir: ${CUDA_DIR}")
else()
    message(STATUS "custom CUDA_DIR is defined as: ${CUDA_DIR}")
endif()

if(NOT TensorRT_DIR)
    set(TensorRT_DIR "/usr/src/tensorrt")
    message(STATUS "TensorRT_DIR is not defined, use default dir: ${TensorRT_DIR}")
else()
    message(STATUS "custom TensorRT_DIR is defined as: ${TensorRT_DIR}")
endif()

if(NOT EXISTS ${CUDA_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${CUDA_DIR} is not exists! Please define -DCUDA_DIR=xxx while TensorRT Backend is enabled.")
endif()

if(NOT EXISTS ${TensorRT_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${TensorRT_DIR} is not exists! Please define -DTensorRT_DIR=xxx while TensorRT Backend is enabled.")
endif()

execute_process(COMMAND sh -c "nm -D libnvinfer.so | grep tensorrt_version"
                WORKING_DIRECTORY /usr/lib/x86_64-linux-gnu
                RESULT_VARIABLE result
                OUTPUT_VARIABLE curr_out
                ERROR_VARIABLE  curr_out)

string(STRIP ${curr_out} TensorRT_Version)
set(TensorRT_Version ${TensorRT_Version} CACHE STRING "TensorRT version" FORCE)

include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

include_directories(${TensorRT_DIR}/include)
# link_directories(${TensorRT_DIR}/lib)

file(GLOB TENSORRT_CORE_SRCS ${CMAKE_SOURCE_DIR}/src/backend/trt/*.cpp)
file(GLOB TENSORRT_CORE_HEAD ${CMAKE_SOURCE_DIR}/src/backend/trt/*.h)