file(GLOB_RECURSE GOMANG_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/backends/trt/*.cpp"
)

add_library(gomang SHARED ${GOMANG_SOURCES})

target_include_directories(gomang
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set_target_properties(gomang PROPERTIES
        VERSION 1.0.0
)

find_package(CUDAToolkit REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer)

target_link_libraries(gomang
        PUBLIC
        ${TENSORRT_LIBRARY}
        CUDA::cudart
        CUDA::cuda_driver
)