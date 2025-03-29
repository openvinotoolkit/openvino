# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

if(OV_COMPILER_IS_INTEL_LLVM)
    # For windows we need to disable warning as error option to make FindSYCL.cmake work
    if (WIN32)
        ov_add_compiler_flags(/WX-)
    endif()

    find_package(IntelSYCL REQUIRED)

    if (WIN32)
        ov_add_compiler_flags(/WX)
        ov_add_compiler_flags(/Wno-ignored-attributes)
    endif()
endif()

if((CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG) AND CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    set(CMAKE_CXX_FLAGS "-Wno-error=deprecated ${CMAKE_CXX_FLAGS}")
endif()

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(OV_COMPILER_IS_CLANG)
    # -Wno-delete-non-abstract-non-virtual-dtor is support > clang 9.0
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 9.0)
        ov_add_compiler_flags(-Wno-delete-non-abstract-non-virtual-dtor)
    else()
        ov_add_compiler_flags(-Wno-delete-non-virtual-dtor)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4267 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ov_add_compiler_flags(/wd4244)
    # '<': signed/unsigned mismatch
    ov_add_compiler_flags(/wd4018)

    # see https://github.com/oneapi-src/oneDNN/issues/2028
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

if(ENABLE_GPU_DEBUG_CAPS)
    add_definitions(-DGPU_DEBUG_CONFIG=1)
    add_definitions(-DENABLE_DEBUG_CAPS=1)
endif()

set(INTEL_GPU_TARGET_OCL_VERSION "300"
  CACHE STRING "Target version of OpenCL which should be used by GPU plugin")

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
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow")
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
