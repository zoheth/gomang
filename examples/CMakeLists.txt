add_executable(test_common test_common.cpp)

target_link_libraries(test_common
        PRIVATE
        gomang
)

target_include_directories(test_common
        PRIVATE
        ${CMAKE_SOURCE_DIR}/gomang
)