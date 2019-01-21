# Copyright (C) 2018 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME perfcheck)

find_package(OpenCV COMPONENTS imgproc QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

file(GLOB SOURCES *.cpp)

add_executable(${TARGET_NAME} ${SOURCES})

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${TARGET_NAME}
        PRIVATE "-Wall"
        PRIVATE "-Wextra"
        PRIVATE "-Wformat"
        PRIVATE "-Wno-missing-field-initializers"
    )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(${TARGET_NAME}
        PRIVATE "-Weverything"
        PRIVATE "-Wno-c++98-compat"
        PRIVATE "-Wno-global-constructors"
        PRIVATE "-Wno-missing-variable-declarations"
        PRIVATE "-Wno-exit-time-destructors"
        PRIVATE "-Wno-undefined-func-template"
    )
endif()

target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
    "${IE_MAIN_SOURCE_DIR}/include"
    "${IE_MAIN_SOURCE_DIR}/samples/common"
    "${IE_MAIN_SOURCE_DIR}/samples/common/format_reader"
    "${IE_MAIN_SOURCE_DIR}/samples/common/os/windows"
    "${CMAKE_SOURCE_DIR}/src/vpu/graph_transformer/include"
    ${OpenCV_INCLUDE_DIRS}
)

add_dependencies(${TARGET_NAME} gflags IE::ie_cpu_extension)
target_link_libraries(${TARGET_NAME} PRIVATE
    ${InferenceEngine_LIBRARIES} format_reader
    IE::ie_cpu_extension
    ${OpenCV_LIBS}
    gflags
    ${CMAKE_DL_LIBS}
)

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FILES" "${CMAKE_CXX_FILES} -fPIE")
