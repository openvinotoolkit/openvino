# Copyright (C) 2018-2026 Intel Corporation
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

  # sources shared by every subset target, regardless of the selected test file(s)
  set(CPU_SUBSET_TEST_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/core_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/skip_tests_config.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/set_device_name.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/cpu_test_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/convolution_params.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/filter_cpu_info.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/fusing_test_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/properties_test.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/quantization_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_fake_quantize.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/transformations/insert_requantize.hpp
    ${CPU_SUBSET_TEST_ABS_PATH})

  if(ARM OR AARCH64)
    list(APPEND CPU_SUBSET_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/arm/filter_cpu_info.cpp)
  elseif(RISCV64)
    list(APPEND CPU_SUBSET_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/riscv64/filter_cpu_info.cpp)
  elseif(X86_64)
    list(APPEND CPU_SUBSET_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/utils/x64/filter_cpu_info.cpp)
  endif()

  ov_add_test_target(
    NAME ${SUBSET_TARGET_NAME}
    SOURCES ${CPU_SUBSET_TEST_SRCS}
    INCLUDES ${INCLUDES}
    DEFINES ${DEFINES}
    DEPENDENCIES ${DEPENDENCIES}
    LINK_LIBRARIES ${LINK_LIBRARIES}
    LABELS OV CPU
  )

  ov_set_threading_interface_for(${SUBSET_TARGET_NAME})
endif()
