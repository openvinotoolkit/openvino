# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: fix in tests
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

set(TARGET_NAME "ov_gpu_unit_tests")

file(GLOB_RECURSE SOURCES_MAIN
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    # openvino graph transformation
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/transformations/*.hpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/transformations/*.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/variable_state.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/multi_tensor_variable_state.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/remote_context.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/remote_tensor.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/usm_host_tensor.cpp"
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/plugin/common_utils.cpp"
  )

if (NOT ENABLE_ONEDNN_FOR_GPU)
    set(EXCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onednn/")
    foreach (SOURCE_FILE IN LISTS SOURCES_MAIN)
        if (SOURCE_FILE MATCHES "${EXCLUDE_DIR}.*")
            list (REMOVE_ITEM SOURCES_MAIN ${SOURCE_FILE})
        endif ()
    endforeach()
endif()

if (MSVC)
  file(GLOB SOURCES_NATVIS
    "${CMAKE_CURRENT_SOURCE_DIR}/float16.natvis"
  )
endif()


set(SOURCES_ALL
    ${SOURCES_MAIN}
    ${SOURCES_NATVIS}
    ${TEST_COMMON_SOURCE_FILES}
  )

add_executable(${TARGET_NAME} ${SOURCES_ALL})
target_compile_definitions(${TARGET_NAME} PRIVATE CI_BUILD_NUMBER="")

ov_set_threading_interface_for(${TARGET_NAME})

# Workaround to avoid warnings during LTO build
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_graph
                                             OpenCL::OpenCL
                                             gtest
                                             gtest_main
                                             gflags
                                             common_test_utils
                                             openvino::reference
                                             gmock)

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_kernels,INTERFACE_INCLUDE_DIRECTORIES>
                                                  $<TARGET_PROPERTY:openvino_intel_gpu_runtime,INTERFACE_INCLUDE_DIRECTORIES>
                                                  ${CMAKE_HOME_DIRECTORY}/src/core/reference/include/
                                                  ${TEST_COMMON_INCLUDE_DIR})

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

if(ENABLE_SSE42)
  ov_sse42_optimization_flags(sse4_2_flags)
  set_source_files_properties(${SOURCES_ALL} PROPERTIES COMPILE_FLAGS "${sse4_2_flags}")
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL)
