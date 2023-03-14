# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# to limit test scope to just a single test file
# improves debugging expirience
  # relative path to specifc test file
if(DEFINED ENABLE_CPU_SUBSET_TESTS_PATH)
  set(SUBSET_TARGET_NAME ov_cpu_func_tests_subset)

  set(CPU_SUBSET_TEST_ABS_PATH)
  set(CPU_SUBSET_TEST_DIR)

  foreach(TEST_PATH ${ENABLE_CPU_SUBSET_TESTS_PATH})
    list(APPEND CPU_SUBSET_TEST_ABS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_PATH})
    get_filename_component(TEST_DIR ${CPU_SUBSET_TEST_ABS_PATH} DIRECTORY)
    list(APPEND CPU_SUBSET_TEST_DIR ${TEST_DIR})
  endforeach()

  # exclude every other test file
  set(EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST
    ${CMAKE_CURRENT_SOURCE_DIR}/behavior
    ${CMAKE_CURRENT_SOURCE_DIR}/bfloat16
    ${CMAKE_CURRENT_SOURCE_DIR}/blob
    ${CMAKE_CURRENT_SOURCE_DIR}/extension
    ${CMAKE_CURRENT_SOURCE_DIR}/onnx
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
    ${CPU_SUBSET_TEST_ABS_PATH})

  addIeTargetTest(
    NAME ${SUBSET_TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES ${INCLUDES}
    EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST}
    OBJECT_FILES ${REQUIRED_OBJECT_FILES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    ADD_CPPLINT
    LABELS
    CPU
  )

  set_ie_threading_interface_for(${SUBSET_TARGET_NAME})
endif()
