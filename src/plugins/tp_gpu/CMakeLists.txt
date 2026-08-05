# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(NOT ENABLE_TP_GPU)
    return()
endif()

set(TARGET_NAME openvino_tp_gpu_plugin)

# Generate embedded OpenCL kernels header at configure time.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernels.cmake)

# ---------------------------------------------------------------------------
# openvino_tp_gpu_ops -- the collective ops (TPAllReduce & co).
#
# The GPU plugin has to know these types: it casts to them in its op factory
# and needs their typeinfo/vtable symbols.  Exposing them as a proper static
# library keeps that dependency declared and reviewable instead of having
# intel_gpu compile TP sources of its own.
# ---------------------------------------------------------------------------
file(GLOB OPS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/op/*.cpp)
file(GLOB OPS_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/tp_gpu/op/*.hpp)

add_library(openvino_tp_gpu_ops STATIC ${OPS_SOURCES} ${OPS_HEADERS})
add_library(openvino::tp_gpu_ops ALIAS openvino_tp_gpu_ops)

target_include_directories(openvino_tp_gpu_ops
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(openvino_tp_gpu_ops PUBLIC openvino::runtime)
set_target_properties(openvino_tp_gpu_ops PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

ov_add_clang_format_target(openvino_tp_gpu_ops_clang FOR_SOURCES ${OPS_SOURCES} ${OPS_HEADERS})

file(GLOB_RECURSE SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
list(REMOVE_ITEM SOURCES ${OPS_SOURCES})
file(GLOB_RECURSE HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
list(REMOVE_ITEM HEADERS ${OPS_HEADERS})

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
target_link_libraries(${TARGET_NAME} PRIVATE
    Threads::Threads
    openvino::runtime::dev   # transformations/utils/utils.hpp
    openvino::tp_gpu_ops
    openvino::zero_loader
    OpenCL::OpenCL)

ov_set_threading_interface_for(${TARGET_NAME})

# must be called after all target_link_libraries
ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(samples)
