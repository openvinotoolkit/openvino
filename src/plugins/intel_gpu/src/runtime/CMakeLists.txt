# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_runtime")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
if(GPU_BACKEND STREQUAL "L0")
    set(GPU_BACKEND "L0" CACHE STRING "selected GPU backed (OCL/L0)" FORCE)
    find_package(LevelZero REQUIRED)
endif()


file(GLOB LIBRARY_SOURCES_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

file(GLOB LIBRARY_SOURCES_OCL
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ocl/*.cpp"
  )

file(GLOB LIBRARY_SOURCES_ZE
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ze/*.cpp"
  )

set(LIBRARY_SOURCES_ALL
    ${LIBRARY_SOURCES_MAIN}
    ${LIBRARY_SOURCES_OCL}
    ${LIBRARY_SOURCES_ZE}
  )

add_library(${TARGET_NAME} STATIC ${LIBRARY_SOURCES_ALL})

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${INCLUDE_DIR}>)

target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

if(COMMAND add_cpplint_target)
  add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endif()

if(GPU_BACKEND STREQUAL "L0")
  target_compile_definitions(${TARGET_NAME} PUBLIC GPU_ENABLE_ZE_BACKEND)
endif()

if(COMMAND set_ie_threading_interface_for)
  set_ie_threading_interface_for(${TARGET_NAME})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    OpenCL
    openvino::itt
    inference_engine_plugin_api
    ${LevelZero_LIBRARY}
  )

if(ENABLE_ONEDNN_FOR_GPU)
  target_link_libraries(${TARGET_NAME} PUBLIC onednn_gpu_tgt)
endif()

if(GPU_ENABLE_ZE_BACKEND)
  target_link_libraries(${TARGET_NAME} PUBLIC LevelZero)
endif()

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} gpu)
