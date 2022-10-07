# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: fix in tests
if(SUGGEST_OVERRIDE_SUPPORTED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(/wd4305)
endif()

set(TARGET_NAME "clDNN_unit_tests64")

file(GLOB_RECURSE SOURCES_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    # ngraph graph transformation
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/transformations/*.hpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/transformations/*.cpp"
  )

if (MSVC)
  file(GLOB SOURCES_NATVIS
    "${CMAKE_CURRENT_SOURCE_DIR}/float16.natvis"
  )
endif()

set(SOURCES_ALL
    ${SOURCES_MAIN}
    ${SOURCES_NATVIS}
  )

add_executable(${TARGET_NAME} ${SOURCES_ALL})

set_ie_threading_interface_for(${TARGET_NAME})

# Workaround to avoid warnings during LTO build
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph
                                             inference_engine
                                             OpenCL
                                             gtest
                                             gtest_main
                                             gflags
                                             ngraph_reference
                                             inference_engine_transformations
                                             gmock)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_kernels,INTERFACE_INCLUDE_DIRECTORIES>
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_runtime,INTERFACE_INCLUDE_DIRECTORIES>
                                                  ${CMAKE_HOME_DIRECTORY}/src/core/reference/include/)

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

ie_sse42_optimization_flags(sse4_2_flags)
set_source_files_properties(${SOURCES_ALL} PROPERTIES COMPILE_FLAGS "${sse4_2_flags}")

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
