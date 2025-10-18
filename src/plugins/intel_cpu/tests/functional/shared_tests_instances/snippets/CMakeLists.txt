# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  add.cpp
  check_broadcast.cpp
  codegen_gelu.cpp
  conv_eltwise.cpp
  edge_replace.cpp
  eltwise_two_results.cpp
  enforce_precision.cpp
  exp.cpp
  fake_quantize_decomposition_test.cpp
  fully_connected.cpp
  gated_mlp.cpp
  group_normalization.cpp
  matmul.cpp
  max_num_params_eltwise.cpp
  mha.cpp
  mha_extracted_reshape.cpp
  mha_fma.cpp
  mha_quantized.cpp
  mha_select.cpp
  mha_split_dim_m.cpp
  mha_transposed_b.cpp
  mha_with_dyn_mul.cpp
  mha_wo_transpose.cpp
  mlp_seq.cpp
  precision_propagation_convertion.cpp
  reduce.cpp
  select.cpp
  softmax.cpp
  three_inputs_eltwise.cpp
  transpose.cpp
  transpose_matmul.cpp
  transpose_softmax.cpp
  two_inputs_and_outputs.cpp
)
list(TRANSFORM current_dir_sources PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

get_property(all_cpu_functional_test_sources GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES)
list(APPEND all_cpu_functional_test_sources ${current_dir_sources})
set_property(GLOBAL PROPERTY OV_CPU_FUNCTIONAL_TEST_SOURCES "${all_cpu_functional_test_sources}")

# Architecture-specific subdirectories
if(X86_64)
  add_subdirectory(x64)
endif()
if(ARM OR AARCH64)
  add_subdirectory(arm)
endif()