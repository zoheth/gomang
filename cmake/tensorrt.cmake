set(CUDA_DIR "" CACHE PATH
        "Path to CUDA installation directory (required for TensorRT backend)")
set(TensorRT_DIR "" CACHE PATH
        "Path to TensorRT installation directory (required for TensorRT backend)")

if(NOT CUDA_DIR)
    set(CUDA_DIR "/usr/local/cuda")
    message(STATUS "[gomang] Using default CUDA directory: ${CUDA_DIR}")
else()
    message(STATUS "[gomang] Using custom CUDA directory: ${CUDA_DIR}")
endif()

if(NOT TensorRT_DIR)
    set(TensorRT_DIR "/usr/src/tensorrt")
    message(STATUS "[gomang] Using default TensorRT directory: ${TensorRT_DIR}")
else()
    message(STATUS "[gomang] Using custom TensorRT directory: ${TensorRT_DIR}")
endif()

if(NOT EXISTS ${CUDA_DIR})
    message(FATAL_ERROR
            "[gomang] CUDA directory not found: ${CUDA_DIR}\n"
            "Please specify correct path using -DCUDA_DIR=<path>")
endif()

if(NOT EXISTS ${TensorRT_DIR})
    message(FATAL_ERROR
            "[gomang] TensorRT directory not found: ${TensorRT_DIR}\n"
            "Please specify correct path using -DTensorRT_DIR=<path>")
endif()

execute_process(
        COMMAND sh -c "nm -D libnvinfer.so | grep tensorrt_version"
        WORKING_DIRECTORY /usr/lib/x86_64-linux-gnu
        RESULT_VARIABLE TENSORRT_VERSION_RESULT
        OUTPUT_VARIABLE TENSORRT_VERSION_OUTPUT
        ERROR_VARIABLE  TENSORRT_VERSION_ERROR
)

if(TENSORRT_VERSION_RESULT EQUAL 0)
    string(STRIP "${TENSORRT_VERSION_OUTPUT}" TensorRT_Version)
    set(TensorRT_Version ${TensorRT_Version} CACHE STRING "TensorRT version" FORCE)
    message(STATUS "[gomang] Detected TensorRT version: ${TensorRT_Version}")
else()
    message(WARNING "[gomang] Failed to detect TensorRT version: ${TENSORRT_VERSION_ERROR}")
endif()

include_directories(
        ${CUDA_DIR}/include
        ${TensorRT_DIR}/include
)

link_directories(${CUDA_DIR}/lib64)