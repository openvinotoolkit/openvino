# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
  file(GLOB LIST_OF_TEST_FILES ${TEST_DIR}/*.cpp)

  # exclude every other test file inside directory
  set(EXCLUDED_SOURCE_PATHS_FOR_TEST
    ${TEST_DIR})

  # list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_utils/fusing_test_utils.cpp
  )

  # create targed for each test file in directory
  foreach(TEST_FILE ${LIST_OF_TEST_FILES})
    # test file name without extension
    get_filename_component(TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_FILE_WE})

    # create target
    addIeTargetTest(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_FILE}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      ADD_CPPLINT
      LABELS
      CPU
    )

    set_ie_threading_interface_for(${TEST_TARGET_NAME})
    # avoid building binaries for every test in case target 'all' is used
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
      EXCLUDE_FROM_ALL ON)
  endforeach()
endfunction()

if(ENABLE_CPU_SPECIFIC_TARGET_PER_TEST)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/subgraph_tests/src ov_cpu_func_subgraph)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/single_layer_tests ov_cpu_func_slt)
endif()

# examples of targets:
# - ov_cpu_func_subgraph_mha
# - ov_cpu_func_slt_convolution
