# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
#exclude every other test file inside directory
  set(EXCLUDED_SOURCE_PATHS_FOR_TEST
    ${TEST_DIR})

#list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
  )

if(X86_64)
    list(APPEND REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
elseif(ARM OR AARCH64)
    list(APPEND REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
elseif(RISCV64)
    list(APPEND REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
endif()

  file(GLOB LIST_OF_TEST_FILES ${TEST_DIR}/*.cpp)
  # create targed for each test file in directory
  foreach(TEST_FILE ${LIST_OF_TEST_FILES})
    # test file name without extension
    get_filename_component(TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_FILE_WE})

    # create target
    ov_add_test_target(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_FILE}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      ADD_CPPLINT
      LABELS OV CPU
    )

    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    # avoid building binaries for every test in case target 'all' is used
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
      EXCLUDE_FROM_ALL ON)
  endforeach()

  # New way of collecting source files for a test target
  # caused by re-organization of test files
  file(GLOB LIST_OF_TEST_CLASSES ${TEST_DIR} ${TEST_DIR}/classes/*.cpp)
  foreach(TEST_CLASS_FILE ${LIST_OF_TEST_CLASSES})
    get_filename_component(TEST_CLASS ${TEST_CLASS_FILE} NAME_WE)
    get_filename_component(TEST_CLASS_FILE_NAME ${TEST_CLASS_FILE} NAME)

    # find all the source files with the name of a class file
    if(X86_64)
        file(GLOB_RECURSE LIST_OF_TEST_ARCH_INSTANCES ${TEST_DIR}/instances/x64/${TEST_CLASS_FILE_NAME})
    elseif(ARM OR AARCH64)
        file(GLOB_RECURSE LIST_OF_TEST_ARCH_INSTANCES ${TEST_DIR}/instances/arm/${TEST_CLASS_FILE_NAME})
    endif()
    file(GLOB_RECURSE LIST_OF_TEST_COMMON_INSTANCES ${TEST_DIR}/instances/common/${TEST_CLASS_FILE_NAME})
    set(LIST_OF_TEST_INSTANCES ${LIST_OF_TEST_COMMON_INSTANCES} ${LIST_OF_TEST_ARCH_INSTANCES})

    set(TEST_INSTANCES "${LIST_OF_TEST_INSTANCES}")
    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_CLASS})

    # create target
    ov_add_test_target(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_CLASS_FILE} ${TEST_INSTANCES}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      LABELS OV CPU
    )

    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    # avoid building binaries for every test in case target 'all' is used
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
      EXCLUDE_FROM_ALL ON)
  endforeach()

endfunction()

if(ENABLE_CPU_SPECIFIC_TARGET_PER_TEST)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common ov_cpu_func_subgraph)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests ov_cpu_func_slt)
endif()

# examples of targets:
# - ov_cpu_func_subgraph_mha
# - ov_cpu_func_slt_convolution
