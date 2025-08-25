# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  broadcast_eltwise_eliminated.cpp
  concat_resize_concat.cpp
  constant_result.cpp
  conv_eltwise_fusion.cpp
  convert_pad_to_group_conv.cpp
  conv_strides_opt.cpp
  gather_weights_decompression.cpp
  get_output_before_activation.cpp
  integer_reduce_mean.cpp
  lora_pattern.cpp
  matmul_const_transposes_extraction.cpp
  matmul_multiply_fusion.cpp
  matmul_squeeze_add.cpp
  memory_LSTMCell.cpp
  mul_conv_fusion.cpp
  multiple_LSTMCell.cpp
  multiply_add.cpp
  mvn_multiply_add.cpp
  parameter_result.cpp
  perm_conv_perm_concat.cpp
  preprocess.cpp
  quantized_convolution_backprop_data.cpp
  quantized_convolution_batch_norm.cpp
  quantized_group_convolution_backprop_data.cpp
  quantized_group_convolution.cpp
  quantized_mat_mul.cpp
  range_add.cpp
  relu_shape_of.cpp
  reshape_squeeze_reshape_relu.cpp
  rotary_pos_emb.cpp
  shared_matmul_gather_weights_decompression.cpp
  shared_matmul_weights_decompression.cpp
  simple_if.cpp
  split_concat_memory.cpp
  split_conv_concat.cpp
  stateful_model.cpp
  transpose_matmul_fusion.cpp
  variadic_split_pad.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")