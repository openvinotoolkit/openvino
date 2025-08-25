# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
  # exclude every other test file inside directory (we pass explicit OBJECT_FILES)
  set(EXCLUDED_SOURCE_PATHS_FOR_TEST ${TEST_DIR})

  get_property(REQUIRED_OBJECT_FILES GLOBAL PROPERTY OV_CPU_FUNC_REQUIRED_OBJECTS)


  # 1) Create target per direct .cpp under TEST_DIR
  set(LIST_OF_TEST_FILES)
  if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
    get_property(LIST_OF_TEST_FILES GLOBAL PROPERTY OV_CPU_FUNC_SLT_SOURCES)
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src")
    get_property(LIST_OF_TEST_FILES GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_SRC_TOP_SOURCES)
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common")
    get_property(LIST_OF_TEST_FILES GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_COMMON_SOURCES)
  endif()

  foreach(TEST_FILE IN LISTS LIST_OF_TEST_FILES)
    get_filename_component(TEST_FILE_WE ${TEST_FILE} NAME_WE)
    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_FILE_WE})
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
      LABELS OV CPU)
    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
  endforeach()

  set(LIST_OF_TEST_CLASSES)
  if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
    get_property(LIST_OF_TEST_CLASSES GLOBAL PROPERTY OV_CPU_FUNC_SLT_CLASS_SOURCES)
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src")
    get_property(LIST_OF_TEST_CLASSES GLOBAL PROPERTY OV_CPU_FUNC_SUBGRAPH_CLASS_SOURCES)
  endif()

  foreach(TEST_CLASS_FILE IN LISTS LIST_OF_TEST_CLASSES)
    get_filename_component(TEST_CLASS ${TEST_CLASS_FILE} NAME_WE)
    get_filename_component(TEST_CLASS_FILE_NAME ${TEST_CLASS_FILE} NAME)
    set(TEST_INSTANCES)
    if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
      set(common_inst ${TEST_DIR}/instances/common/${TEST_CLASS_FILE_NAME})
      if(EXISTS ${common_inst})
        list(APPEND TEST_INSTANCES ${common_inst})
      endif()
      if(X86_64)
        set(arch_inst ${TEST_DIR}/instances/x64/${TEST_CLASS_FILE_NAME})
      elseif(ARM OR AARCH64)
        set(arch_inst ${TEST_DIR}/instances/arm/${TEST_CLASS_FILE_NAME})
      elseif(RISCV64)
        set(arch_inst ${TEST_DIR}/instances/riscv64/${TEST_CLASS_FILE_NAME})
      endif()
      if(DEFINED arch_inst AND EXISTS ${arch_inst})
        list(APPEND TEST_INSTANCES ${arch_inst})
      endif()
    endif()

    set(TEST_TARGET_NAME ${TARGET_PREFIX}_${TEST_CLASS})
    ov_add_test_target(
      NAME ${TEST_TARGET_NAME}
      ROOT ${TEST_DIR}
      INCLUDES ${INCLUDES}
      EXCLUDED_SOURCE_PATHS ${EXCLUDED_SOURCE_PATHS_FOR_TEST}
      OBJECT_FILES ${REQUIRED_OBJECT_FILES} ${TEST_CLASS_FILE} ${TEST_INSTANCES}
      DEFINES ${DEFINES}
      DEPENDENCIES ${DEPENDENCIES}
      LINK_LIBRARIES ${LINK_LIBRARIES}
      LABELS OV CPU)
    ov_set_threading_interface_for(${TEST_TARGET_NAME})
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES EXCLUDE_FROM_ALL ON)
  endforeach()

endfunction()

if(ENABLE_CPU_SPECIFIC_TARGET_PER_TEST)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src ov_cpu_func_subgraph)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/common ov_cpu_func_subgraph_common)
  create_target_per_test_for_directory(${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests ov_cpu_func_slt)
endif()

# examples of targets:
# - ov_cpu_func_subgraph_mha
# - ov_cpu_func_slt_convolution
