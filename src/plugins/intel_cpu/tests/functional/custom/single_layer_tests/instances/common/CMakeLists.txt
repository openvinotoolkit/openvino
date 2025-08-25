# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  activation.cpp
  bitwise_shift.cpp
  col2im.cpp
  comparison.cpp
  conversion.cpp
  convolution_backprop_data.cpp
  convolution.cpp
  eltwise.cpp
  extremum.cpp
  interpolate.cpp
  inverse.cpp
  logical.cpp
  matmul.cpp
  multinomial.cpp
  mvn.cpp
  nms_rotated.cpp
  pooling.cpp
  random_uniform.cpp
  reduce.cpp
  scaled_attn.cpp
  segment_max.cpp
  softmax.cpp
  sparse_fill_empty_rows.cpp
  string_tensor_pack.cpp
  string_tensor_unpack.cpp
  transpose.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")