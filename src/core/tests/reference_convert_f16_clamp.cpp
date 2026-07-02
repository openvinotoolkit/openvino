// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/reference/convert.hpp"

// These tests target the clamping conversions used by ConvertPrecision when it demotes
// float constants to f16. The clamp must map *finite* out-of-range values to ±f16::max,
// while preserving IEEE special values (±inf, NaN). The input sizes are chosen so that the
// special values land inside a full vectorized lane (indices 0..7) as well as in the scalar
// tail remainder (last elements), exercising both the JIT/intrinsics and the scalar paths.

using ov::bfloat16;
using ov::float16;

namespace {
constexpr float f16_max = 65504.0f;  // std::numeric_limits<float16>::max()

float f(float16 v) {
    return static_cast<float>(v);
}
}  // namespace

TEST(reference_convert_f16_clamp, f32_to_f16_preserves_specials_and_clamps_finite) {
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();

    // 20 elements => two full 8-wide vector chunks + a 4-element tail.
    // Specials are placed in the first chunk (0..7) and again in the tail (16..19).
    const std::vector<float> in = {pinf, ninf, nan,  1.0f, 2.0f * f16_max, -2.0f * f16_max, 0.0f, f16_max,  // chunk 0
                                   2.0f, 3.0f, 4.0f, 5.0f, 6.0f,           7.0f,            8.0f, 9.0f,      // chunk 1
                                   pinf, ninf, nan,  -3.0f * f16_max};                                      // tail
    std::vector<float16> out(in.size(), float16(0.0f));

    ov::reference::convert_from_f32_to_f16_with_clamp(in.data(), out.data(), in.size());

    EXPECT_TRUE(std::isinf(f(out[0])) && f(out[0]) > 0) << "+inf lost: " << f(out[0]);
    EXPECT_TRUE(std::isinf(f(out[1])) && f(out[1]) < 0) << "-inf lost: " << f(out[1]);
    EXPECT_TRUE(std::isnan(f(out[2]))) << "NaN lost: " << f(out[2]);
    EXPECT_EQ(f(out[3]), 1.0f);
    EXPECT_EQ(out[4], std::numeric_limits<float16>::max()) << "finite overflow must clamp, not become inf";
    EXPECT_EQ(out[5], std::numeric_limits<float16>::lowest()) << "finite underflow must clamp, not become -inf";
    EXPECT_EQ(f(out[6]), 0.0f);
    EXPECT_EQ(out[7], std::numeric_limits<float16>::max());
    // Second chunk: plain finite values round-trip.
    for (size_t i = 8; i < 16; ++i) {
        EXPECT_EQ(f(out[i]), in[i]) << "at index " << i;
    }
    // Tail remainder.
    EXPECT_TRUE(std::isinf(f(out[16])) && f(out[16]) > 0) << "+inf lost in tail: " << f(out[16]);
    EXPECT_TRUE(std::isinf(f(out[17])) && f(out[17]) < 0) << "-inf lost in tail: " << f(out[17]);
    EXPECT_TRUE(std::isnan(f(out[18]))) << "NaN lost in tail: " << f(out[18]);
    EXPECT_EQ(out[19], std::numeric_limits<float16>::lowest());
}

TEST(reference_convert_f16_clamp, bf16_to_f16_preserves_specials_and_clamps_finite) {
    const float pinf = std::numeric_limits<float>::infinity();
    const float ninf = -std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float big = 1.0e5f;  // finite bf16, exceeds f16::max -> must clamp

    const std::vector<bfloat16> in = {bfloat16(pinf), bfloat16(ninf), bfloat16(nan),   bfloat16(1.0f),
                                      bfloat16(big),  bfloat16(-big), bfloat16(0.0f),  bfloat16(f16_max),  // chunk 0
                                      bfloat16(2.0f), bfloat16(3.0f), bfloat16(4.0f),  bfloat16(5.0f),
                                      bfloat16(6.0f), bfloat16(7.0f), bfloat16(8.0f),  bfloat16(9.0f),     // chunk 1
                                      bfloat16(pinf), bfloat16(ninf), bfloat16(nan),   bfloat16(-big)};    // tail
    std::vector<float16> out(in.size(), float16(0.0f));

    ov::reference::convert_from_bf16_to_f16_with_clamp(in.data(), out.data(), in.size());

    EXPECT_TRUE(std::isinf(f(out[0])) && f(out[0]) > 0) << "+inf lost: " << f(out[0]);
    EXPECT_TRUE(std::isinf(f(out[1])) && f(out[1]) < 0) << "-inf lost: " << f(out[1]);
    EXPECT_TRUE(std::isnan(f(out[2]))) << "NaN lost: " << f(out[2]);
    EXPECT_EQ(f(out[3]), 1.0f);
    EXPECT_EQ(out[4], std::numeric_limits<float16>::max()) << "finite overflow must clamp, not become inf";
    EXPECT_EQ(out[5], std::numeric_limits<float16>::lowest()) << "finite underflow must clamp, not become -inf";
    EXPECT_EQ(f(out[6]), 0.0f);
    for (size_t i = 8; i < 16; ++i) {
        EXPECT_EQ(f(out[i]), static_cast<float>(in[i])) << "at index " << i;
    }
    EXPECT_TRUE(std::isinf(f(out[16])) && f(out[16]) > 0) << "+inf lost in tail: " << f(out[16]);
    EXPECT_TRUE(std::isinf(f(out[17])) && f(out[17]) < 0) << "-inf lost in tail: " << f(out[17]);
    EXPECT_TRUE(std::isnan(f(out[18]))) << "NaN lost in tail: " << f(out[18]);
    EXPECT_EQ(out[19], std::numeric_limits<float16>::lowest());
}
