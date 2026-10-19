# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Creates a single per-test target; remaining arguments are the target's sources.
function(ov_cpu_add_per_test_target TEST_TARGET_NAME)
  ov_add_test_target(
    NAME ${TEST_TARGET_NAME}
    SOURCES ${ARGN}
    INCLUDES ${INCLUDES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    LABELS OV CPU
  )

  ov_set_threading_interface_for(${TEST_TARGET_NAME})
  # avoid building binaries for every test in case target 'all' is used
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    EXCLUDE_FROM_ALL ON)
endfunction()

#create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
#list of sources required for each test
  set(COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp
  )

if(X86_64)
    list(APPEND COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
    list(APPEND COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
elseif(RISCV64)
    list(APPEND COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
endif()

  # test files enabled for the current configuration, see sources.cmake
  set(ENABLED_TEST_SRCS ${CPU_FUNC_TESTS_SRCS} ${TMP_EXPLICITLY_ENABLED_TESTS})

  set(LIST_OF_TEST_FILES ${ENABLED_TEST_SRCS})
  list(FILTER LIST_OF_TEST_FILES INCLUDE REGEX "^${TEST_DIR}/[^/]+\\.cpp$")
  # create targed for each test file in directory
  foreach(TEST_FILE IN LISTS LIST_OF_TEST_FILES)
    # test file name without extension
    get_filename_component(TEST_FILE_WE ${TEST_FILE} NAME_WE)

    ov_cpu_add_per_test_target(${TARGET_PREFIX}_${TEST_FILE_WE} ${COMMON_SOURCES} ${TEST_FILE})
  endforeach()

  # New way of collecting source files for a test target
  # caused by re-organization of test files
  set(LIST_OF_TEST_CLASSES ${ENABLED_TEST_SRCS})
  list(FILTER LIST_OF_TEST_CLASSES INCLUDE REGEX "^${TEST_DIR}/classes/[^/]+\\.cpp$")
  foreach(TEST_CLASS_FILE IN LISTS LIST_OF_TEST_CLASSES)
    get_filename_component(TEST_CLASS ${TEST_CLASS_FILE} NAME_WE)
    get_filename_component(TEST_CLASS_FILE_NAME ${TEST_CLASS_FILE} NAME)
    string(REPLACE "." "\\." TEST_CLASS_FILE_NAME_REGEX ${TEST_CLASS_FILE_NAME})

    # find all the source files with the name of a class file;
    # instances of the other architectures are already absent from ENABLED_TEST_SRCS
    set(LIST_OF_TEST_INSTANCES ${ENABLED_TEST_SRCS})
    list(FILTER LIST_OF_TEST_INSTANCES INCLUDE REGEX
      "^${TEST_DIR}/instances/[^/]+/(.*/)?${TEST_CLASS_FILE_NAME_REGEX}$")

    ov_cpu_add_per_test_target(${TARGET_PREFIX}_${TEST_CLASS}
      ${COMMON_SOURCES} ${TEST_CLASS_FILE} ${LIST_OF_TEST_INSTANCES})
  endforeach()

endfunction()

if(ENABLE_CPU_SPECIFIC_TARGET_PER_TEST)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src ov_cpu_func_subgraph)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common ov_cpu_func_subgraph_common)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/arm ov_cpu_func_subgraph_arm)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests ov_cpu_func_slt)
endif()

# examples of targets:
# - ov_cpu_func_subgraph_mha
# - ov_cpu_func_slt_convolution
