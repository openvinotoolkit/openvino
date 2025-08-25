# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  activation.cpp
  col2im.cpp
  conversion.cpp
  convolution_backprop_data.cpp
  convolution.cpp
  eltwise.cpp
  interpolate.cpp
  matmul.cpp
  mvn.cpp
  pooling.cpp
  random_uniform.cpp
  reduce.cpp
  rms_norm.cpp
  softmax.cpp
  transpose.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")