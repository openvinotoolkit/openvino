# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wno-delete-non-abstract-non-virtual-dtor)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4267 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4244)
    # '<': signed/unsigned mismatch
    ov_add_compiler_flags(/wd4018)
endif()

if(ENABLE_GPU_DEBUG_CAPS)
    add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

set(INTEL_GPU_TARGET_OCL_VERSION "200" CACHE STRING "Target version of OpenCL which should be used by GPU plugin")

add_definitions(-DCL_TARGET_OPENCL_VERSION=${INTEL_GPU_TARGET_OCL_VERSION})

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(thirdparty)
include(thirdparty/cmake/rapidjson.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Werror)
endif()

add_subdirectory(src/runtime)
add_subdirectory(src/kernel_selector)
add_subdirectory(src/graph)

file(GLOB_RECURSE PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_gpu/plugin/*.hpp)

ov_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GPU"
              SOURCES ${PLUGIN_SOURCES}
              DEFAULT_CONFIG ${PLUGIN_DEFAULT_CONFIG}
              VERSION_DEFINES_FOR src/plugin/plugin.cpp)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph openvino::pugixml)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/)

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Workaround to avoid warnings during LTO build
if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
