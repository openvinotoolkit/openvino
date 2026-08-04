# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

if(CMAKE_COMPILER_IS_GNUCXX)
    ov_add_compiler_flags(-Wno-strict-aliasing)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
    # see https://github.com/oneapi-src/oneDNN/issues/2028
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
endif()

if(WIN32)
    add_definitions(-DNOMINMAX)
endif()

if(ENABLE_GPU_DEBUG_CAPS)
    add_definitions(-DGPU_DEBUG_CONFIG=1)
    add_definitions(-DENABLE_DEBUG_CAPS=1)
endif()

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
              VERSION_DEFINES_FOR src/plugin/plugin.cpp
              # ADD_CLANG_TIDY makes ov_add_plugin apply clang-tidy to this target.
              # Libraries not created via ov_add_plugin call ov_apply_clang_tidy() directly.
              ADD_CLANG_TIDY)

# Release kernels are built for speed (-O3), not size (-Os), to maximize kernel throughput.
# Measured: l2norm (64 MB) GPU plugin kernel time benefits from -O3 over -Os on AMD/Intel dGPUs.
target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph openvino::pugixml)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/)

ov_set_threading_interface_for(${TARGET_NAME})
ov_gpu_set_runtime_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Workaround to avoid warnings during LTO build
if(CMAKE_COMPILER_IS_GNUCXX)
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow")
endif()

ov_build_target_faster(${TARGET_NAME} PCH)

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# must be called after all target_link_libraries
# ov_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
