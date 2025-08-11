# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(current_dir_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/check_broadcast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/codegen_gelu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv_eltwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/edge_replace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise_two_results.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/enforce_precision.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/exp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fake_quantize_decomposition_test.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/fully_connected.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/gated_mlp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/group_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/max_num_params_eltwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_extracted_reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_fma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_quantized.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_select.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_split_dim_m.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_transposed_b.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_with_dyn_mul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mha_wo_transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mlp_seq.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/precision_propagation_convertion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/select.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/three_inputs_eltwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose_matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transpose_softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/two_inputs_and_outputs.cpp
)

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