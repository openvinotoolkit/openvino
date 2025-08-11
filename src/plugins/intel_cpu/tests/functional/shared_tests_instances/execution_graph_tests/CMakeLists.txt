# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/add_output.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/disable_lowering_precision.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/duplicate_inputs_outputs_names.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/keep_assign.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nms_transformation_for_last_node.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/num_inputs_fusing_bin_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/remove_parameter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/runtime_precision.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")