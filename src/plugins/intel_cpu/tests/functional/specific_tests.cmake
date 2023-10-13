# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# to limit test scope to a particular test files
# improves debugging expirience
# relative path to specifc test files
if(DEFINED ENABLE_CPU_SUBSET_TESTS_PATH)
  set(SUBSET_TARGET_NAME ov_cpu_func_tests_subset)

  set(CPU_SUBSET_TEST_ABS_PATH_LIST)
  set(CPU_SUBSET_TEST_DIR)

  # convert to list to be able to iterate over
  set(CPU_SUBSET_TESTS_PATH_LIST ${ENABLE_CPU_SUBSET_TESTS_PATH})
  separate_arguments(CPU_SUBSET_TESTS_PATH_LIST)

  foreach(TEST_PATH ${CPU_SUBSET_TESTS_PATH_LIST})
    list(APPEND CPU_SUBSET_TEST_ABS_PATH_LIST ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_PATH})
    get_filename_component(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_PATH} DIRECTORY)
    list(APPEND CPU_SUBSET_TEST_DIR ${TEST_DIR})
  endforeach()

  set(CPU_SUBSET_TEST_ABS_PATH "${CPU_SUBSET_TEST_ABS_PATH_LIST}")

  # exclude every other test file
  set(EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/behavior
    ${CMAKE_CURRENT_SOURCE_DIR}/extension
    ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances
    ${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src)

  # list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests/classes/eltwise.cpp
    ${CPU_SUBSET_TEST_ABS_PATH})

  ov_add_test_target(
    NAME ${SUBSET_TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES ${INCLUDES}
    EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST}
    OBJECT_FILES ${REQUIRED_OBJECT_FILES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    LABELS OV CPU
  )

  ov_set_threading_interface_for(${SUBSET_TARGET_NAME})
endif()
