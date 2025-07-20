# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set(snippets_SRC
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
    utils.hpp
)

# Convert to full paths
foreach(src_file ${snippets_SRC})
    list(APPEND snippets_SRC_FULL "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
endforeach()

set_property(GLOBAL APPEND PROPERTY CPU_FUNC_TESTS_SRC ${snippets_SRC_FULL})

# Use priority-based architecture selection to avoid multiple definitions
if(X86_64 OR X86)
    add_subdirectory(x64)
elseif(AARCH64 OR ARM)
    add_subdirectory(arm)
endif()
