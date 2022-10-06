# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT ENABLE_INTEL_GPU)
    return()
endif()

set (TARGET_NAME "openvino_intel_gpu_plugin")

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wno-strict-aliasing -Wno-missing-declarations)
    ie_add_compiler_flags(-msse4.1 -msse4.2)
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
        ie_add_compiler_flags(-Wno-error=terminate)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # 4267 4244 conversion from 'XXX' to 'YYY', possible loss of data
    ie_add_compiler_flags(/wd4244)
    ie_add_compiler_flags(/wd4267)
    # '<': signed/unsigned mismatch
    ie_add_compiler_flags(/wd4018)
endif()

if(ENABLE_GPU_DEBUG_CAPS)
  add_definitions(-DGPU_DEBUG_CONFIG=1)
endif()

set(MAIN_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(thirdparty)
add_subdirectory(src/runtime)
add_subdirectory(src/kernel_selector)
add_subdirectory(src/graph)

file(GLOB_RECURSE PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/plugin/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/include/intel_gpu/plugin/*.hpp)

addVersionDefines(src/plugin/plugin.cpp CI_BUILD_NUMBER)

ie_add_plugin(NAME ${TARGET_NAME}
              DEVICE_NAME "GPU"
              SOURCES ${PLUGIN_SOURCES}
              VERSION_DEFINES_FOR src/plugin/plugin.cpp)

target_compile_options(${TARGET_NAME} PRIVATE
    $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph openvino::pugixml)

target_include_directories(${TARGET_NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/)

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

# Workaround to avoid warnings during LTO build
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized"
                                                  LINK_FLAGS_RELWITHDEBINFO "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

if(ENABLE_TESTS)
  add_subdirectory(tests)
endif()

# Failed because of OpenCL
# ie_add_api_validator_post_build_step(TARGET ${TARGET_NAME})
