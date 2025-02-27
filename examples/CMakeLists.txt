add_executable(test_common test_common.cpp)

target_link_libraries(test_common
        PRIVATE
        gomang
)

target_include_directories(test_common
        PRIVATE
        ${CMAKE_SOURCE_DIR}/gomang
)

if (ENABLE_IREE)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_IREE
    )

    target_include_directories(test_common
            PRIVATE
            ${CMAKE_SOURCE_DIR}/model/iree
    )
endif ()

if (ENABLE_TENSORRT)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_TENSORRT
    )
endif ()

if (ENABLE_MNN)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_MNN
    )
endif ()

if (ENABLE_NCNN)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_NCNN
    )
endif ()


file(GLOB IREE_MODULE_SRC "${CMAKE_SOURCE_DIR}/models/iree/*.c")

add_executable(test_iree iree_test.cpp
        ${IREE_MODULE_SRC}
)

target_link_libraries(test_iree
        PRIVATE
        gomang
)

target_include_directories(test_iree
        PRIVATE
        ${CMAKE_SOURCE_DIR}/models/iree
)