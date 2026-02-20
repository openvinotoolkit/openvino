// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/kernels/scaled_attn/softmax_kernel.hpp"

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

namespace {
TEST(SoftmaxKernelTest, AttnSoftmaxKernelWithSparseMask) {
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> output(input.size(), 0.0f);
    std::vector<uint8_t> sparse_mask = {1, 0, 1, 0};  // Masking some elements, block size 2
    float scale = 1.0f;
    float* alibi = nullptr;
    void* attn_mask = nullptr;
    uint8_t* causal_mask = nullptr;
    bool select_nfltmax_at_0 = false;
    size_t len = input.size();
    size_t total_size = input.size();
    ov::element::Type attn_mask_prec = ov::element::f32;
    ov::element::Type dst_precision = ov::element::f32;
    const float* sink = nullptr;
    float alibi_slope = 0.0f;
    size_t sparse_block_size = 2;
    ov::Extensions::Cpu::XARCH::attn_softmax_kernel<float>(input.data(),
                                                           output.data(),
                                                           scale,
                                                           alibi,
                                                           attn_mask,
                                                           causal_mask,
                                                           select_nfltmax_at_0,
                                                           len,
                                                           total_size,
                                                           attn_mask_prec,
                                                           dst_precision,
                                                           sink,
                                                           alibi_slope,
                                                           sparse_mask.data(),
                                                           sparse_block_size);
    std::vector<float> expect_output = {0.00483724f, 0.013149f, 0.0f, 0.0f, 0.264104f, 0.71791f, 0.0f, 0.0f};
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expect_output[i], 1e-5f);
    }
}

TEST(SoftmaxKernelTest, AttnSoftmaxKernelWithNaNInputAndSparseMask) {
    std::vector<float> input = {1.0f, 2.0f, std::nanf(""), 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> output(input.size(), 0.0f);
    std::vector<uint8_t> sparse_mask = {1, 0, 1, 0};  // Masking some elements, block size 2
    float scale = 1.0f;
    float* alibi = nullptr;
    void* attn_mask = nullptr;
    uint8_t* causal_mask = nullptr;
    bool select_nfltmax_at_0 = false;
    size_t len = input.size();
    size_t total_size = input.size();
    ov::element::Type attn_mask_prec = ov::element::f32;
    ov::element::Type dst_precision = ov::element::f32;
    const float* sink = nullptr;
    float alibi_slope = 0.0f;
    size_t sparse_block_size = 2;
    ov::Extensions::Cpu::XARCH::attn_softmax_kernel<float>(input.data(),
                                                           output.data(),
                                                           scale,
                                                           alibi,
                                                           attn_mask,
                                                           causal_mask,
                                                           select_nfltmax_at_0,
                                                           len,
                                                           total_size,
                                                           attn_mask_prec,
                                                           dst_precision,
                                                           sink,
                                                           alibi_slope,
                                                           sparse_mask.data(),
                                                           sparse_block_size);
    std::vector<float> expect_output = {0.00483724f, 0.013149f, 0.0f, 0.0f, 0.264104f, 0.71791f, 0.0f, 0.0f};
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_NEAR(output[i], expect_output[i], 1e-5f);
    }
}

}  // namespace
