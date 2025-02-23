file(GLOB_RECURSE CORE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
)

if (ENABLE_TENSORRT)
    file(GLOB_RECURSE TRT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/trt/*.cpp")
endif ()

if (ENABLE_MNN)
    file(GLOB_RECURSE MNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/mnn/*.cpp")
endif ()

add_library(gomang SHARED
        ${CORE_SOURCES}
        ${TRT_SOURCES}
        ${MNN_SOURCES}
)

target_include_directories(gomang
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

set_target_properties(gomang PROPERTIES
        VERSION 1.0.0
        SOVERSION 1
)

if (ENABLE_TENSORRT)
    find_package(CUDAToolkit REQUIRED)
    find_library(TENSORRT_LIBRARY nvinfer)

    target_link_libraries(gomang PUBLIC
            ${TENSORRT_LIBRARY}
            CUDA::cudart
            CUDA::cuda_driver
    )
endif ()

if (ENABLE_MNN)
    target_link_libraries(gomang PUBLIC MNN)
endif ()