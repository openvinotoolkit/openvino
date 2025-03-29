# Copyright (C) 2018-2025 Intel Corporation
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
    ${CMAKE_CURRENT_SOURCE_DIR}/custom
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances)

  # list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CPU_SUBSET_TEST_ABS_PATH})

if(NOT (ARM OR AARCH64))
  list(APPEND EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm)
endif()
if(NOT RISCV64)
  list(APPEND EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64)
endif()
if(NOT X86_64)
  list(APPEND EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64)
endif()

  ov_add_test_target(
    NAME ${SUBSET_TARGET_NAME}
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}
    INCLUDES ${INCLUDES}
    EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_SUBSET_TEST} ${EXCLUDED_SOURCE_PATHS}
    OBJECT_FILES ${REQUIRED_OBJECT_FILES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    LABELS OV CPU
  )

  ov_set_threading_interface_for(${SUBSET_TARGET_NAME})
endif()
