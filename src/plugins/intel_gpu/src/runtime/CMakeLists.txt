# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_runtime")

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

set(LIBRARY_SOURCES_ALL
    ${LIBRARY_SOURCES_MAIN}
    ${LIBRARY_SOURCES_OCL}
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

if(COMMAND set_ie_threading_interface_for)
  set_ie_threading_interface_for(${TARGET_NAME})
endif()

target_link_libraries(${TARGET_NAME} PRIVATE
    OpenCL
    openvino::itt
    inference_engine_plugin_api
  )

if(ENABLE_ONEDNN_FOR_GPU)
  target_link_libraries(${TARGET_NAME} PUBLIC onednn_gpu_tgt)
endif()

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} gpu)
