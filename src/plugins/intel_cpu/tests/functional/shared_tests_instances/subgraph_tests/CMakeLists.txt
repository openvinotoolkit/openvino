# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/broadcast_eltwise_eliminated.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/concat_resize_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constant_result.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_eltwise_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/convert_pad_to_group_conv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_strides_opt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gather_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/get_output_before_activation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/integer_reduce_mean.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lora_pattern.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_const_transposes_extraction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_multiply_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul_squeeze_add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/memory_LSTMCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mul_conv_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multiple_LSTMCell.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/multiply_add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mvn_multiply_add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parameter_result.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/perm_conv_perm_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/preprocess.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_convolution_backprop_data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_convolution_batch_norm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_group_convolution_backprop_data.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_group_convolution.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/quantized_mat_mul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/range_add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/relu_shape_of.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reshape_squeeze_reshape_relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rotary_pos_emb.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/shared_matmul_gather_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/shared_matmul_weights_decompression.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/simple_if.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/split_concat_memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/split_conv_concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/stateful_model.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose_matmul_fusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/variadic_split_pad.cpp
)

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")