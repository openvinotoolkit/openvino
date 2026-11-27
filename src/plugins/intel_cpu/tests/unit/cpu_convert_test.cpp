// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/common/cpu_convert.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

using namespace ov::intel_cpu;

namespace {

template <typename T>
void expect_to_boolean(ov::element::Type src_prc, const std::vector<T>& src, const std::vector<uint8_t>& expected) {
    ASSERT_EQ(src.size(), expected.size());

    std::vector<uint8_t> dst_seq(src.size(), 0xAA);
    cpu_convert(src.data(), dst_seq.data(), src_prc, ov::element::boolean, src.size());
    EXPECT_EQ(dst_seq, expected);

    std::vector<uint8_t> dst_par(src.size(), 0xAA);
    cpu_parallel_convert(src.data(), dst_par.data(), src_prc, ov::element::boolean, src.size());
    EXPECT_EQ(dst_par, expected);
}

}  // namespace

TEST(CpuConvertTests, NumericToBooleanFloat32) {
    const std::vector<float> src = {0.0f,
                                    -0.0f,
                                    0.25f,
                                    -0.25f,
                                    1.0f,
                                    -1.0f,
                                    42.5f,
                                    -42.5f,
                                    std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()};
    const std::vector<uint8_t> expected = {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    expect_to_boolean(ov::element::f32, src, expected);
}

TEST(CpuConvertTests, NumericToBooleanFloat16) {
    std::vector<ov::float16> src;
    std::vector<uint8_t> expected;
    const std::vector<float> pattern = {0.0f, -0.0f, 0.25f, -0.25f, 1.0f, -5.0f};
    const std::vector<uint8_t> pattern_expected = {0, 0, 1, 1, 1, 1};
    // Exercise multiple internal conversion batches.
    for (int repeat = 0; repeat < 60; ++repeat) {
        for (size_t i = 0; i < pattern.size(); ++i) {
            src.emplace_back(pattern[i]);
            expected.push_back(pattern_expected[i]);
        }
    }
    expect_to_boolean(ov::element::f16, src, expected);
}

TEST(CpuConvertTests, NumericToBooleanBfloat16) {
    const std::vector<ov::bfloat16> src = {
        ov::bfloat16(0.0f),
        ov::bfloat16(0.25f),
        ov::bfloat16(-0.25f),
        ov::bfloat16(-3.0f),
    };
    const std::vector<uint8_t> expected = {0, 1, 1, 1};
    expect_to_boolean(ov::element::bf16, src, expected);
}

TEST(CpuConvertTests, NumericToBooleanInt32) {
    const std::vector<int32_t> src = {
        0,
        1,
        -1,
        42,
        -42,
        std::numeric_limits<int32_t>::min(),
        std::numeric_limits<int32_t>::max(),
    };
    const std::vector<uint8_t> expected = {0, 1, 1, 1, 1, 1, 1};
    expect_to_boolean(ov::element::i32, src, expected);
}

TEST(CpuConvertTests, NumericToBooleanInt8Uint8Int64) {
    expect_to_boolean<int8_t>(ov::element::i8, {0, 1, -1, -128, 127}, {0, 1, 1, 1, 1});
    expect_to_boolean<uint8_t>(ov::element::u8, {0, 1, 2, 255}, {0, 1, 1, 1});
    expect_to_boolean<int64_t>(ov::element::i64,
                               {0, 1, -1, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()},
                               {0, 1, 1, 1, 1});
}
