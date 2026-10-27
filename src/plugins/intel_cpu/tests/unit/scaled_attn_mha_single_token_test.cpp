// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_quant.hpp"
#include "nodes/kernels/scaled_attn/mha_single_token.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/plain_tensor.hpp"
#include "utils/precision_support.h"

namespace {

using ov::Extensions::Cpu::XARCH::mha_single_token;
using ov::Extensions::Cpu::XARCH::attn_quant_by_token;
using ov::intel_cpu::CpuParallel;
using ov::intel_cpu::PlainTensor;
using ov::intel_cpu::TbbPartitioner;

constexpr size_t kHeadSize = 64;

void run_zero_point_case(float min_key, float max_key, bool expect_fp16_zero_point_overflow) {
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        GTEST_SKIP() << "The host has no native FP16 support.";
    }

    std::array<ov::float16, kHeadSize> query_data{};
    std::array<ov::float16, kHeadSize> key_data{};
    std::array<ov::float16, kHeadSize> value_data{};
    std::array<uint8_t, kHeadSize> key_cache{};
    std::array<uint8_t, kHeadSize> value_cache{};
    std::array<float, 2> key_scale_zp{};
    std::array<float, 2> value_scale_zp{};

    for (size_t i = 0; i < kHeadSize; ++i) {
        query_data[i] = ov::float16(i % 2 == 0 ? 0.001F : -0.001F);
        key_data[i] = ov::float16(i % 2 == 0 ? max_key : min_key);
        value_data[i] = ov::float16(1.0F);
    }

    PlainTensor key_input;
    key_input.resize<ov::float16>({1, 1, 1, kHeadSize}, key_data.data());
    PlainTensor value_input;
    value_input.resize<ov::float16>({1, 1, 1, kHeadSize}, value_data.data());
    PlainTensor present_key;
    present_key.resize<uint8_t>({1, 1, 1, kHeadSize}, key_cache.data());
    PlainTensor present_value;
    present_value.resize<uint8_t>({1, 1, 1, kHeadSize}, value_cache.data());
    PlainTensor key_params;
    key_params.resize<float>({1, 1, 1, 2}, key_scale_zp.data());
    PlainTensor value_params;
    value_params.resize<float>({1, 1, 1, 2}, value_scale_zp.data());
    auto cpu_parallel = std::make_shared<CpuParallel>(TbbPartitioner::STATIC);

    attn_quant_by_token(key_input, present_key, key_params, 0, kHeadSize, cpu_parallel);
    attn_quant_by_token(value_input, present_value, value_params, 0, kHeadSize, cpu_parallel);

    ASSERT_NEAR(key_scale_zp[0], (max_key - min_key) / 255.0F, 1.0e-6F);
    ASSERT_NEAR(key_scale_zp[1], -min_key / key_scale_zp[0], 1.0F);
    for (size_t i = 0; i < kHeadSize; ++i) {
        ASSERT_EQ(key_cache[i], i % 2 == 0 ? 255 : 0);
    }
    ASSERT_EQ(std::isfinite(static_cast<float>(ov::float16(key_scale_zp[1]))),
              !expect_fp16_zero_point_overflow);
    ASSERT_FLOAT_EQ(value_scale_zp[0], 0.05F);
    ASSERT_FLOAT_EQ(value_scale_zp[1], -20.0F);
    for (const auto value : value_cache) {
        ASSERT_EQ(value, 0);
    }

    PlainTensor query;
    query.resize<ov::float16>({1, 1, 1, kHeadSize}, query_data.data());

    PlainTensor output;
    output.resize<ov::float16>({1, 1, 1, kHeadSize});
    PlainTensor attention_weights;
    attention_weights.resize<float>({1, 1, 1, 16});
    PlainTensor attention_scores;
    PlainTensor head_sum;
    PlainTensor alibi_mask;
    PlainTensor attention_mask;
    PlainTensor beams;
    PlainTensor sink_input;
    mha_single_token(query,
                     present_key,
                     present_value,
                     alibi_mask,
                     attention_mask,
                     beams,
                     output,
                     attention_weights,
                     attention_scores,
                     false,
                     false,
                     0.0F,
                     key_params,
                     value_params,
                     head_sum,
                     kHeadSize,
                     kHeadSize,
                     false,
                     sink_input,
                     cpu_parallel);

    for (size_t i = 0; i < kHeadSize; ++i) {
        const auto expected_f16 = ov::float16((static_cast<float>(value_cache[i]) - value_scale_zp[1]) *
                                              value_scale_zp[0]);
        const float expected = static_cast<float>(expected_f16);
        const float actual = static_cast<float>(output.ptr<ov::float16>()[i]);
        EXPECT_TRUE(std::isfinite(actual)) << "nonfinite output at feature " << i;
        EXPECT_NEAR(actual, expected, 0.02F) << "mismatch at feature " << i;
    }
}

TEST(ScaledAttnMhaSingleTokenTest, Fp16QueryWithU8CacheKeepsOutputFiniteForLargeZeroPoint) {
    run_zero_point_case(-60000.0F, -59968.0F, true);
    run_zero_point_case(59968.0F, 60000.0F, true);
    run_zero_point_case(-1000.0F, -968.0F, false);
}

}  // namespace
