file(GLOB_RECURSE CORE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp"
)

if (ENABLE_IREE)
    file(GLOB_RECURSE IREE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/backends/iree/*.cpp")
    file(GLOB IREE_MODULE_SRC "${CMAKE_SOURCE_DIR}/models/iree/*.c")
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
        ${IREE_MODULE_SRC}
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
#            iree::runtime
            iree_base_base
            iree_hal_hal
            iree_hal_drivers_local_task_registration_registration
            iree_hal_local_loaders_vmvx_module_loader
            iree_modules_hal_hal
            iree_vm_vm
            iree_vm_bytecode_module
    )

    target_include_directories(gomang
            PRIVATE
            ${CMAKE_SOURCE_DIR}/models/iree
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