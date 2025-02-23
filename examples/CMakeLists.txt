add_executable(test_common test_common.cpp)

target_link_libraries(test_common
        PRIVATE
        gomang
)

target_include_directories(test_common
        PRIVATE
        ${CMAKE_SOURCE_DIR}/gomang
)

if(ENABLE_TENSORRT)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_TENSORRT
    )
endif()

if(ENABLE_MNN)
    target_compile_definitions(test_common
            PUBLIC
            ENABLE_MNN
    )
endif ()