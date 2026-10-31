// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/bfloat16.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>

#include "common_test_utils/float_util.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/log.hpp"

namespace ov::test {

template <typename T>
std::string to_hex(T value) {
    std::stringstream ss;
    ss << "0x" << std::hex << std::setw(sizeof(T) * 2) << std::setfill('0') << value;
    return ss.str();
}

//***********************
// NOTE
//***********************
// This test uses exact comparisons of floating point values. It is testing for bit-exact
// creation and truncation/rounding of bfloat16 values.
TEST(bfloat16, conversions) {
    bfloat16 bf;
    const char* source_string;
    std::string bf_string;

    // 1.f, the ground-truth value
    source_string = "0  01111111  000 0000";
    bf = ov::test::utils::bits_to_bfloat16(source_string);
    EXPECT_EQ(bf, bfloat16(1.0));
    bf_string = ov::test::utils::bfloat16_to_bits(bf);
    EXPECT_STREQ(source_string, bf_string.c_str());

    // 1.03125f, the exact upper bound
    source_string = "0  01111111  000 0100";
    bf = ov::test::utils::bits_to_bfloat16(source_string);
    EXPECT_EQ(bf, bfloat16(1.03125));
    bf_string = ov::test::utils::bfloat16_to_bits(bf);
    EXPECT_STREQ(source_string, bf_string.c_str());
}

TEST(bfloat16, round_to_nearest) {
    const char* fstring;
    std::string expected;
    float fvalue;
    uint16_t bf_round;

    fstring = "0  01111111  000 0100 1000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_round, 0x3F85);

    fstring = "0  01111111  000 0100 0000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_round, 0x3F84);

    fstring = "0  01111111  111 1111 1000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_round, 0x4000);

    // 1.9921875f, the next representable number which should not round up
    fstring = "0  01111111  111 1111 0000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest(fvalue);
    EXPECT_EQ(bf_round, 0x3FFF);
}

TEST(bfloat16, round_to_nearest_even) {
    const char* fstring;
    float fvalue;
    uint16_t bf_round;

    fstring = "0  01111111  000 0100 1000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x3F84);

    fstring = "0  01111111  000 0101 1000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x3F86);

    fstring = "0  01111111  000 0101 0000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x3F85);

    // Just above the halfway point must round up.
    fstring = "0  01111111  000 0000 1000 0000 0000 0001";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x3F81);

    fstring = "0  01111111  111 1111 1000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x4000);

    fstring = "0  01111111  111 1111 0000 0000 0000 0000";
    fvalue = ov::test::utils::bits_to_float(fstring);
    bf_round = bfloat16::round_to_nearest_even(fvalue);
    EXPECT_EQ(bf_round, 0x3FFF);
}

struct BFloat16RoundingTestCase {
    std::string name;
    uint32_t input;
    uint16_t expected;
};

class BFloat16RoundingTest : public ::testing::TestWithParam<BFloat16RoundingTestCase> {};

INSTANTIATE_TEST_SUITE_P(
    BoundaryAndSpecialValues,
    BFloat16RoundingTest,
    ::testing::Values(BFloat16RoundingTestCase{"positive_below_even_tie", 0x3f807fffU, 0x3f80},
                      BFloat16RoundingTestCase{"positive_even_tie", 0x3f808000U, 0x3f80},
                      BFloat16RoundingTestCase{"positive_above_even_tie", 0x3f808001U, 0x3f81},
                      BFloat16RoundingTestCase{"positive_odd_tie", 0x3f818000U, 0x3f82},
                      BFloat16RoundingTestCase{"negative_below_even_tie", 0xbf807fffU, 0xbf80},
                      BFloat16RoundingTestCase{"negative_even_tie", 0xbf808000U, 0xbf80},
                      BFloat16RoundingTestCase{"negative_above_even_tie", 0xbf808001U, 0xbf81},
                      BFloat16RoundingTestCase{"negative_odd_tie", 0xbf818000U, 0xbf82},
                      BFloat16RoundingTestCase{"positive_below_overflow_tie", 0x7f7f7fffU, 0x7f7f},
                      BFloat16RoundingTestCase{"positive_overflow_tie", 0x7f7f8000U, 0x7f80},
                      BFloat16RoundingTestCase{"negative_below_overflow_tie", 0xff7f7fffU, 0xff7f},
                      BFloat16RoundingTestCase{"negative_overflow_tie", 0xff7f8000U, 0xff80},
                      BFloat16RoundingTestCase{"positive_subnormal_below_tie", 0x00007fffU, 0x0000},
                      BFloat16RoundingTestCase{"positive_subnormal_tie", 0x00008000U, 0x0000},
                      BFloat16RoundingTestCase{"positive_subnormal_above_tie", 0x00008001U, 0x0001},
                      BFloat16RoundingTestCase{"negative_subnormal_below_tie", 0x80007fffU, 0x8000},
                      BFloat16RoundingTestCase{"negative_subnormal_tie", 0x80008000U, 0x8000},
                      BFloat16RoundingTestCase{"negative_subnormal_above_tie", 0x80008001U, 0x8001},
                      BFloat16RoundingTestCase{"positive_infinity", 0x7f800000U, 0x7f80},
                      BFloat16RoundingTestCase{"negative_infinity", 0xff800000U, 0xff80},
                      BFloat16RoundingTestCase{"positive_zero", 0x00000000U, 0x0000},
                      BFloat16RoundingTestCase{"negative_zero", 0x80000000U, 0x8000}),
    [](const testing::TestParamInfo<BFloat16RoundingTestCase>& info) {
        return info.param.name;
    });

TEST_P(BFloat16RoundingTest, round_to_nearest_even_matches_expected_bits) {
    const auto& test_case = GetParam();
    float value;
    std::memcpy(&value, &test_case.input, sizeof(value));
    EXPECT_EQ(bfloat16::round_to_nearest_even(value), test_case.expected);
    EXPECT_EQ(bfloat16(value).to_bits(), test_case.expected);
}

TEST(bfloat16, random_finite_rounding_matches_nearest_neighbor_reference) {
    std::mt19937 rng(0xB16F16);
    std::uniform_int_distribution<uint32_t> magnitude_distribution(0, 0x7f7effffU);
    const auto as_float = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };

    for (size_t i = 0; i < 4096; ++i) {
        const uint32_t input_bits = magnitude_distribution(rng) | (rng() & 0x80000000U);
        const uint32_t lower_bits = input_bits & 0xffff0000U;
        const uint32_t upper_bits = lower_bits + 0x00010000U;

        const double input = as_float(input_bits);
        const double lower_distance = std::fabs(input - static_cast<double>(as_float(lower_bits)));
        const double upper_distance = std::fabs(input - static_cast<double>(as_float(upper_bits)));

        const bool choose_upper =
            upper_distance < lower_distance || (upper_distance == lower_distance && ((lower_bits >> 16) & 1U));
        const uint16_t expected_bits = static_cast<uint16_t>((choose_upper ? upper_bits : lower_bits) >> 16);

        EXPECT_EQ(bfloat16::round_to_nearest_even(as_float(input_bits)), expected_bits)
            << "input bits 0x" << std::hex << input_bits;
    }
}

struct BFloat16NaNTestCase {
    std::string name;
    const char* bits;
    uint16_t expected;
};

class BFloat16NaNTest : public ::testing::TestWithParam<BFloat16NaNTestCase> {};

INSTANTIATE_TEST_SUITE_P(
    NaNBitPatterns,
    BFloat16NaNTest,
    ::testing::Values(
        // Signaling NaNs whose payload is entirely in the discarded low bits.
        BFloat16NaNTestCase{"positive_signaling_low_payload", "0  11111111  000 0000 0000 0000 0000 0001", 0x7FC0},
        BFloat16NaNTestCase{"negative_signaling_low_payload", "1  11111111  000 0000 0000 0000 0000 0001", 0xFFC0},
        // NaN mantissas near the sign boundary must not carry into sign/exponent.
        BFloat16NaNTestCase{"positive_nan_near_sign_boundary", "0  11111111  111 1111 1000 0000 0000 0000", 0x7FFF},
        BFloat16NaNTestCase{"negative_nan_near_sign_boundary", "1  11111111  111 1111 1000 0000 0000 0000", 0xFFFF},
        // Quiet NaNs retain their sign and the payload bits that fit in BF16.
        BFloat16NaNTestCase{"positive_quiet_payload", "0  11111111  100 0001 0010 0011 0100 0101", 0x7FC1},
        BFloat16NaNTestCase{"negative_quiet_payload", "1  11111111  100 0001 0010 0011 0100 0101", 0xFFC1}),
    [](const testing::TestParamInfo<BFloat16NaNTestCase>& info) {
        return info.param.name;
    });

TEST_P(BFloat16NaNTest, round_to_nearest_even_preserves_quiet_nan) {
    const auto& test_case = GetParam();
    const auto value = ov::test::utils::bits_to_float(test_case.bits);
    const auto rounded = bfloat16::round_to_nearest_even(value);

    EXPECT_EQ(rounded, test_case.expected);
    EXPECT_EQ(rounded & 0x7F80U, 0x7F80U) << "expected a NaN exponent";
    EXPECT_NE(rounded & 0x007FU, 0U) << "expected a non-zero NaN mantissa";
    EXPECT_NE(rounded & 0x0040U, 0U) << "expected a quiet NaN";
    EXPECT_EQ(bfloat16(value).to_bits(), test_case.expected);
}

TEST(bfloat16, to_float) {
    bfloat16 bf;
    const char* source_string;

    // 1.f, the ground-truth value
    source_string = "0  01111111  000 0000";
    bf = ov::test::utils::bits_to_bfloat16(source_string);
    float f = static_cast<float>(bf);
    EXPECT_EQ(f, 1.0f);

    // 1.03125f, the exact upper bound
    source_string = "0  01111111  000 0100";
    bf = ov::test::utils::bits_to_bfloat16(source_string);
    f = static_cast<float>(bf);
    EXPECT_EQ(f, 1.03125f);
}

TEST(bfloat16, numeric_limits) {
    bfloat16 infinity = std::numeric_limits<bfloat16>::infinity();
    bfloat16 neg_infinity = -std::numeric_limits<bfloat16>::infinity();
    bfloat16 quiet_nan = std::numeric_limits<bfloat16>::quiet_NaN();
    bfloat16 signaling_nan = std::numeric_limits<bfloat16>::signaling_NaN();

    // Would be nice if we could have bfloat16 overloads for these, but it would require adding
    // overloads into ::std. So we just cast to float here. We can't rely on an implicit cast
    // because it fails with some versions of AppleClang.
    EXPECT_TRUE(std::isinf(static_cast<float>(infinity)));
    EXPECT_TRUE(std::isinf(static_cast<float>(neg_infinity)));
    EXPECT_TRUE(std::isnan(static_cast<float>(quiet_nan)));
    EXPECT_TRUE(std::isnan(static_cast<float>(signaling_nan)));
}

TEST(benchmark, bfloat16) {
    size_t buffer_size = 128 * 3 * 224 * 224;
    ov::AlignedBuffer data(buffer_size * sizeof(float), 4096);
    float* f = static_cast<float*>(data.get_ptr());
    // vector<float> data(buffer_size);
    std::mt19937 rng(2112);
    std::uniform_real_distribution<float> distribution(-300, 300);
    for (size_t i = 0; i < buffer_size; ++i) {
        f[i] = distribution(rng);
    }
    OPENVINO_INFO("buffer size ", buffer_size, " floats or ", data.size(), " bytes");

    {
        ov::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        for (size_t i = 0; i < buffer_size; ++i) {
            p[i] = bfloat16(f[i]);
        }
    }

    {
        ov::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        for (size_t i = 0; i < buffer_size; ++i) {
            p[i] = bfloat16::truncate(f[i]);
        }
    }

    {
        ov::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        for (size_t i = 0; i < buffer_size; ++i) {
            p[i] = bfloat16::round_to_nearest(f[i]);
        }
    }

    {
        ov::AlignedBuffer bf_data(buffer_size * sizeof(bfloat16), 4096);
        bfloat16* p = static_cast<bfloat16*>(bf_data.get_ptr());
        for (size_t i = 0; i < buffer_size; ++i) {
            p[i] = bfloat16::round_to_nearest_even(f[i]);
        }
    }
}

TEST(bfloat16, assigns) {
    bfloat16 bf16;
    bf16 = 2.0;
    EXPECT_EQ(bf16, float16(2.0));

    std::vector<float> f32vec{1.0, 2.0, 4.0};
    std::vector<bfloat16> bf16vec;
    std::copy(f32vec.begin(), f32vec.end(), std::back_inserter(bf16vec));
    for (size_t i = 0; i < f32vec.size(); ++i) {
        EXPECT_EQ(f32vec.at(i), bf16vec.at(i));
    }

    f32vec = {-1.0, -2.0, -3.0};
    for (size_t i = 0; i < f32vec.size(); ++i) {
        bf16vec[i] = f32vec[i];
    }
    for (size_t i = 0; i < f32vec.size(); ++i) {
        EXPECT_EQ(f32vec.at(i), bf16vec.at(i));
    }

    float f32arr[] = {1.0, 2.0, 4.0};
    bfloat16 bf16arr[sizeof(f32arr)];
    for (size_t i = 0; i < sizeof(f32arr) / sizeof(f32arr[0]); ++i) {
        bf16arr[i] = f32arr[i];
        EXPECT_EQ(f32arr[i], bf16arr[i]);
    }
}

TEST(bfloat16, operators) {
    bfloat16 a(2.0);
    bfloat16 b(3.5);
    bfloat16 c(5.5);
    bfloat16 d(7.0);
    ASSERT_TRUE(a + b == c);
    ASSERT_TRUE(a == c - b);
    ASSERT_TRUE(a * b == d);
    ASSERT_TRUE(a == d / b);
}

TEST(bfloat16, to_string) {
    const bfloat16 bf16{1.5f};
    EXPECT_EQ(std::to_string(bf16), std::to_string(1.5f));
}

TEST(bfloat16, size) {
    const bfloat16 bf16{10.12f};
    EXPECT_EQ(sizeof(bf16), 2);
}
}  // namespace ov::test
