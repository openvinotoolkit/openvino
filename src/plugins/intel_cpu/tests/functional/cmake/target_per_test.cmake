# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#create targed with prefix TARGET_PREFIX for each test file in directory TEST_DIR
function(create_target_per_test_for_directory TEST_DIR TARGET_PREFIX)
  # exclude every other test file inside directory (we pass explicit OBJECT_FILES)
  set(EXCLUDED_SOURCE_PATHS_FOR_TEST ${TEST_DIR})

  # list of object files required for each test
  set(REQUIRED_OBJECT_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp)

  if(X86_64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
  elseif(ARM OR AARCH64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
  elseif(RISCV64)
    list(APPEND REQUIRED_OBJECT_FILES ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
  endif()


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

  # 2) Class-based targets with instances
  set(LIST_OF_TEST_CLASSES)
  if(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests")
    set(LIST_OF_TEST_CLASSES
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/activation.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/bitwise_shift.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/col2im.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/comparison.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/conversion.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution_backprop_data.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/convolution.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/eltwise.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/extremum.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/interpolate.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/logical.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/matmul.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/mvn.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/pooling.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/random_uniform.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/reduce.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/rms_norm.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/scaled_attn.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/segment_max.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/softmax.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/sparse_fill_empty_rows.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_pack.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/string_tensor_unpack.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/single_layer_tests/classes/transpose.cpp)
  elseif(TEST_DIR STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src")
    set(LIST_OF_TEST_CLASSES
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/concat_sdp.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_concat.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/conv_maxpool_activ.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/custom/subgraph_tests/src/classes/eltwise_chain.cpp)
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
