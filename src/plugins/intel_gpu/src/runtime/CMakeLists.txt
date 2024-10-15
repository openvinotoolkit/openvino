# Copyright (C) 2018-2024 Intel Corporation
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

file(GLOB_RECURSE SYCL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/ocl/sycl_*.cpp")

if(NOT OV_COMPILER_IS_INTEL_LLVM)
    list(REMOVE_ITEM LIBRARY_SOURCES_ALL ${SYCL_SOURCES})
endif()

add_library(${TARGET_NAME} STATIC ${LIBRARY_SOURCES_ALL})

if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions(${TARGET_NAME} PUBLIC OPENVINO_STATIC_LIBRARY)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${INCLUDE_DIR}>)

target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CONFIG:Release>:
    $<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os
    >
  >
  $<$<CXX_COMPILER_ID:GNU>:
    $<$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,6.0>:
      -Wno-ignored-attributes
    >
  >)

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

if(OV_COMPILER_IS_INTEL_LLVM)
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SYCL_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_SYCL)
endif()

ov_set_threading_interface_for(${TARGET_NAME})

target_link_libraries(${TARGET_NAME} PRIVATE
    OpenCL::OpenCL
    openvino::itt
    openvino::runtime::dev
  )

if(ENABLE_ONEDNN_FOR_GPU)
  target_link_libraries(${TARGET_NAME} PUBLIC onednn_gpu_tgt)
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})
