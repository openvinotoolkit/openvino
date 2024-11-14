# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_intel_gpu_graph")

file(GLOB_RECURSE SYCL_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/impls/sycl/*")
file(GLOB_RECURSE LIBRARY_SRC
    "${INCLUDE_DIR}/*.h"
    "${INCLUDE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  )

if (NOT ENABLE_ONEDNN_FOR_GPU)
    set(EXCLUDE_DIR "/onednn/")
    foreach (SOURCE_FILE ${LIBRARY_SRC})
        string (FIND ${SOURCE_FILE} ${EXCLUDE_DIR} EXCLUDE_DIR_FOUND)
        if (NOT ${EXCLUDE_DIR_FOUND} EQUAL -1)
            list (REMOVE_ITEM LIBRARY_SRC ${SOURCE_FILE})
        endif ()
    endforeach(SOURCE_FILE)
endif()

if(NOT OV_COMPILER_IS_INTEL_LLVM)
    list(REMOVE_ITEM LIBRARY_SRC ${SYCL_SOURCES})
endif()

add_library(${TARGET_NAME} STATIC ${LIBRARY_SRC})

target_include_directories(${TARGET_NAME} PUBLIC
                                          $<BUILD_INTERFACE:${MAIN_DIR}/src>
                                          $<BUILD_INTERFACE:${INCLUDE_DIR}>
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                          $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)

target_compile_options(${TARGET_NAME} PRIVATE
  $<$<CONFIG:Release>:$<IF:$<CXX_COMPILER_ID:MSVC>,/Os,-Os>>)

if(OV_COMPILER_IS_INTEL_LLVM)
    add_sycl_to_target(TARGET ${TARGET_NAME} SOURCES ${SYCL_SOURCES})
    target_compile_definitions(${TARGET_NAME} PUBLIC OV_GPU_WITH_SYCL)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC OpenCL::OpenCL openvino::shape_inference)
target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_kernels
                                             openvino_intel_gpu_runtime
                                             openvino::itt
                                             openvino::reference
                                             openvino::runtime::dev
                                             openvino::runtime)

if(ENABLE_ONEDNN_FOR_GPU)
    target_link_libraries(${TARGET_NAME} PUBLIC onednn_gpu_tgt)
endif()

if(COMMAND add_cpplint_target)
  add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endif()

ov_set_threading_interface_for(${TARGET_NAME})

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

if(ENABLE_SSE42)
  ov_sse42_optimization_flags(sse4_2_flags)
  set_source_files_properties(impls/cpu/detection_output.cpp PROPERTIES
    COMPILE_FLAGS "${sse4_2_flags}"
    COMPILE_DEFINITIONS "HAVE_SSE")
endif()
