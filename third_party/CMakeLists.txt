if (ENABLE_MNN)
    set(MNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/MNN)
    if (NOT EXISTS ${MNN_ROOT})
        message(FATAL_ERROR "MNN submodule not found!")
    endif ()

    set(MNN_INCLUDE_DIR ${MNN_ROOT}/include)

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(MNN_LIBRARY_DIR ${MNN_ROOT}/lib/Debug)
    else()
        set(MNN_LIBRARY_DIR ${MNN_ROOT}/lib/Release)
    endif()

    find_library(MNN_LIBRARY
            NAMES MNN libMNN
            PATHS ${MNN_LIBRARY_DIR}
            NO_DEFAULT_PATH
    )

    if(NOT MNN_LIBRARY)
        message(FATAL_ERROR "MNN library not found in ${MNN_LIBRARY_DIR}")
    endif()

    add_library(MNN INTERFACE)
    target_include_directories(MNN
            INTERFACE
            ${MNN_INCLUDE_DIR}
    )
    target_link_libraries(MNN
            INTERFACE
            ${MNN_LIBRARY}
    )
endif ()