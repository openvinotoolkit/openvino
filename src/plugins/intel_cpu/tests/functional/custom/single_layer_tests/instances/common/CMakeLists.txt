# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/activation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bitwise_shift.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/col2im.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/comparison.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conversion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution_backprop_data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convolution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extremum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/interpolate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inverse.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/logical.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multinomial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mvn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/nms_rotated.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pooling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/random_uniform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/scaled_attn.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/segment_max.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse_fill_empty_rows.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/string_tensor_pack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/string_tensor_unpack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")