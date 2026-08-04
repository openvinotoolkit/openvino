# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_TP_GPU)
    return()
endif()

set(TARGET_NAME openvino_tp_gpu_plugin)

# Generate embedded OpenCL kernels header at configure time.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernels.cmake)

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "TP_GPU"
              PSEUDO_DEVICE
              SOURCES ${SOURCES} ${HEADERS}
              VERSION_DEFINES_FOR src/version.cpp
              ADD_CLANG_FORMAT)

target_include_directories(${TARGET_NAME}
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR}/generated)

ov_build_target_faster(${TARGET_NAME}
    UNITY
)

find_package(Threads REQUIRED)
find_package(OpenCL REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads openvino::zero_loader OpenCL::OpenCL)

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(samples)
