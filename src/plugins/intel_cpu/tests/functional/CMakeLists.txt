# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME ov_cpu_func_tests)

add_library(cpuSpecificRtInfo STATIC
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.hpp
    $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src/utils/rt_info/memory_formats_attribute.cpp)
target_link_libraries(cpuSpecificRtInfo PRIVATE openvino::runtime)

set(INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} $<TARGET_PROPERTY:openvino_intel_cpu_plugin,SOURCE_DIR>/src)
set(DEPENDENCIES openvino_intel_cpu_plugin template_extension)
set(LINK_LIBRARIES funcSharedTests cpuSpecificRtInfo openvino::snippets ov_snippets_models)

if(ENABLE_OV_ONNX_FRONTEND)
    list(APPEND DEFINES TEST_MODELS="${TEST_MODEL_ZOO}")
else()
    set(EXCLUDED_SOURCE_PATHS ${CMAKE_CURRENT_SOURCE_DIR}/extension ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/onnx)
endif()

if(NOT (ARM OR AARCH64))
    list(APPEND EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/instances/arm
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src/arm
        ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/arm)
else()
    # temporary disable all custom tests for ARM
    list(APPEND EXCLUDED_SOURCE_PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
        ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests)
    # except ones which already enabled
    file(GLOB_RECURSE TMP_LIST_OF_TEST_CLASSES          ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/classes/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_COMMON_TEST_INSTANCES ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/instances/common/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_ARM_TEST_INSTANCES    ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/instances/arm/*.cpp)
    file(GLOB_RECURSE TMP_LIST_OF_ARM_SUBGRAPH_TESTS    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/arm/*.cpp)
    list(APPEND TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS
        ${TMP_LIST_OF_TEST_CLASSES} ${TMP_LIST_OF_COMMON_TEST_INSTANCES} ${TMP_LIST_OF_ARM_TEST_INSTANCES} ${TMP_LIST_OF_ARM_SUBGRAPH_TESTS})
    set(TMP_EXPLICITLY_ENABLED_TESTS "${TMP_LIST_OF_EXPLICITLY_ENABLED_TESTS}")
endif()

if(NOT X86_64)
    list(APPEND EXCLUDED_SOURCE_PATHS
         ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/instances/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src/x64
         ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/x64)
endif()

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        INCLUDES ${INCLUDES}
        EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS}
        OBJECT_FILES ${TMP_EXPLICITLY_ENABLED_TESTS}
        DEFINES ${DEFINES}
        DEPENDENCIES ${DEPENDENCIES}
        LINK_LIBRARIES ${LINK_LIBRARIES}
        ADD_CPPLINT
        LABELS CPU
)

ov_set_threading_interface_for(${TARGET_NAME})

include(specific_tests.cmake)
include(target_per_test.cmake)

# exclude 'ov_cpu_func_tests' from target 'all' if specific tests path was specified
if(DEFINED ENABLE_CPU_SPECIFIC_TESTS_PATH)
    set_target_properties(${TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
endif()
