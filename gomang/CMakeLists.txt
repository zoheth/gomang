file(GLOB_RECURSE CORE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
)

if (ENABLE_IREE)
    file(GLOB_RECURSE IREE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/iree/*.cpp")
endif ()

if (ENABLE_TENSORRT)
    file(GLOB_RECURSE TRT_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/trt/*.cpp")
endif ()

if (ENABLE_MNN)
    file(GLOB_RECURSE MNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/mnn/*.cpp")
endif ()

if (ENABLE_NCNN)
    file(GLOB_RECURSE NCNN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/ncnn/*.cpp")
endif ()

add_library(gomang SHARED
        ${CORE_SOURCES}
        ${IREE_SOURCES}
        ${TRT_SOURCES}
        ${MNN_SOURCES}
        ${NCNN_SOURCES}
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

if(ENABLE_IREE)
    target_link_libraries(gomang PUBLIC
            iree::vm
            # HAL runtime
            iree::hal
            # Base utilities
            iree::base
            # Specific HAL drivers you enabled
            iree::hal::drivers::vulkan

            iree::runtime  # 主运行时库
            iree::modules::hal
    )
endif ()

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

if (ENABLE_NCNN)
    target_link_libraries(gomang PUBLIC ncnn)
endif ()