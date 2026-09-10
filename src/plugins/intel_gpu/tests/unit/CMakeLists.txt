# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# TODO: fix in tests
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-suggest-override")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ov_add_compiler_flags(/wd4305)
endif()

if(OV_GPU_WITH_SYCL AND UNIX)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld")
  string(REPLACE "-pie" "" CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
endif()

set(TARGET_NAME "ov_gpu_unit_tests")

include(${CMAKE_CURRENT_SOURCE_DIR}/sources.cmake)

# Files excluded from the completeness check when their corresponding runtime/backend condition
# is not active for this build configuration - mirrors the conditional list(APPEND) logic above.
set(GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES)
if(NOT ENABLE_ONEDNN_FOR_GPU)
    list(APPEND GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${GPU_UNIT_TESTS_ONEDNN_SRCS})
endif()
if(NOT GPU_RT_TYPE STREQUAL "OCL")
    list(APPEND GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${GPU_UNIT_TESTS_OCL_RT_SRCS})
endif()
if(NOT GPU_RT_TYPE STREQUAL "ZE")
    list(APPEND GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${GPU_UNIT_TESTS_ZE_RT_SRCS})
endif()
if(NOT GPU_RT_TYPE STREQUAL "SYCL")
    list(APPEND GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES ${GPU_UNIT_TESTS_SYCL_RT_SRCS})
endif()

if (MSVC)
  set(SOURCES_NATVIS "${CMAKE_CURRENT_SOURCE_DIR}/float16.natvis")
endif()

ov_add_test_target(
    NAME
        ${TARGET_NAME}
    SOURCES
        ${GPU_UNIT_TESTS_SRCS}
        ${SOURCES_NATVIS}
        ${TEST_COMMON_SOURCE_FILES}
    DEFINES
        CI_BUILD_NUMBER=""
    INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/
        $<TARGET_PROPERTY:openvino_intel_gpu_kernels,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:openvino_intel_gpu_runtime,INTERFACE_INCLUDE_DIRECTORIES>
        ${CMAKE_HOME_DIRECTORY}/src/core/reference/include/
        ${TEST_COMMON_INCLUDE_DIR}
    LINK_LIBRARIES
        openvino_intel_gpu_graph
        gtest
        gtest_main
        gflags
        common_test_utils
        openvino::reference
        gmock
    LABELS
        OV UNIT GPU
    CHECK_SOURCES_LISTED
    CHECK_SOURCES_EXCLUDE_FILES
        ${GPU_UNIT_TESTS_CHECK_SOURCES_EXCLUDE_FILES}
)

ov_set_threading_interface_for(${TARGET_NAME})
ov_gpu_set_runtime_interface_for(${TARGET_NAME})

# Workaround to avoid warnings during LTO build
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS_RELEASE "-Wno-error=maybe-uninitialized -Wno-maybe-uninitialized -Wno-stringop-overflow")
endif()

set_target_properties(${TARGET_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ${ENABLE_LTO})

if(WIN32)
  target_link_libraries(${TARGET_NAME} PRIVATE setupapi)
elseif((NOT ANDROID) AND (UNIX))
  target_link_libraries(${TARGET_NAME} PRIVATE pthread)
endif()

if(ENABLE_SSE42)
  ov_sse42_optimization_flags(sse4_2_flags)
  target_compile_options(${TARGET_NAME} PRIVATE ${sse4_2_flags})
endif()

ov_build_target_faster(${TARGET_NAME} PCH)
