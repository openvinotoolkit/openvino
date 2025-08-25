# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  add_output.cpp
  disable_lowering_precision.cpp
  duplicate_inputs_outputs_names.cpp
  keep_assign.cpp
  nms_transformation_for_last_node.cpp
  num_inputs_fusing_bin_conv.cpp
  remove_parameter.cpp
  runtime_precision.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")