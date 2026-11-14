// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/bfloat16.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "utils/cpp/bit_cast.hpp"

namespace ov::intel_cpu::test {

struct IntelCpuBFloat16RoundingTestCase {
    std::string name;
    uint32_t input;
    uint16_t expected;
};

class IntelCpuBFloat16RoundingTest : public ::testing::TestWithParam<IntelCpuBFloat16RoundingTestCase> {};

INSTANTIATE_TEST_SUITE_P(
    BoundaryAndSpecialValues,
    IntelCpuBFloat16RoundingTest,
    ::testing::Values(IntelCpuBFloat16RoundingTestCase{"positive_below_even_tie", 0x3f807fffU, 0x3f80},
                      IntelCpuBFloat16RoundingTestCase{"positive_even_tie", 0x3f808000U, 0x3f80},
                      IntelCpuBFloat16RoundingTestCase{"positive_above_even_tie", 0x3f808001U, 0x3f81},
                      IntelCpuBFloat16RoundingTestCase{"positive_odd_tie", 0x3f818000U, 0x3f82},
                      IntelCpuBFloat16RoundingTestCase{"negative_below_even_tie", 0xbf807fffU, 0xbf80},
                      IntelCpuBFloat16RoundingTestCase{"negative_even_tie", 0xbf808000U, 0xbf80},
                      IntelCpuBFloat16RoundingTestCase{"negative_above_even_tie", 0xbf808001U, 0xbf81},
                      IntelCpuBFloat16RoundingTestCase{"negative_odd_tie", 0xbf818000U, 0xbf82},
                      IntelCpuBFloat16RoundingTestCase{"positive_below_overflow_tie", 0x7f7f7fffU, 0x7f7f},
                      IntelCpuBFloat16RoundingTestCase{"positive_overflow_tie", 0x7f7f8000U, 0x7f80},
                      IntelCpuBFloat16RoundingTestCase{"negative_below_overflow_tie", 0xff7f7fffU, 0xff7f},
                      IntelCpuBFloat16RoundingTestCase{"negative_overflow_tie", 0xff7f8000U, 0xff80},
                      IntelCpuBFloat16RoundingTestCase{"positive_subnormal_below_tie", 0x00007fffU, 0x0000},
                      IntelCpuBFloat16RoundingTestCase{"positive_subnormal_tie", 0x00008000U, 0x0000},
                      IntelCpuBFloat16RoundingTestCase{"positive_subnormal_above_tie", 0x00008001U, 0x0001},
                      IntelCpuBFloat16RoundingTestCase{"negative_subnormal_below_tie", 0x80007fffU, 0x8000},
                      IntelCpuBFloat16RoundingTestCase{"negative_subnormal_tie", 0x80008000U, 0x8000},
                      IntelCpuBFloat16RoundingTestCase{"negative_subnormal_above_tie", 0x80008001U, 0x8001},
                      IntelCpuBFloat16RoundingTestCase{"positive_infinity", 0x7f800000U, 0x7f80},
                      IntelCpuBFloat16RoundingTestCase{"negative_infinity", 0xff800000U, 0xff80},
                      IntelCpuBFloat16RoundingTestCase{"positive_zero", 0x00000000U, 0x0000},
                      IntelCpuBFloat16RoundingTestCase{"negative_zero", 0x80000000U, 0x8000}),
    [](const testing::TestParamInfo<IntelCpuBFloat16RoundingTestCase>& info) {
        return info.param.name;
    });

TEST_P(IntelCpuBFloat16RoundingTest, rounds_boundaries_and_special_values) {
    const auto& test_case = GetParam();
    float input;
    std::memcpy(&input, &test_case.input, sizeof(input));
    EXPECT_EQ(bfloat16_t(input).to_bits(), test_case.expected);
}

struct IntelCpuBFloat16NaNTestCase {
    std::string name;
    uint32_t input;
    uint16_t expected;
};

class IntelCpuBFloat16NaNTest : public ::testing::TestWithParam<IntelCpuBFloat16NaNTestCase> {};

INSTANTIATE_TEST_SUITE_P(
    NaNBitPatterns,
    IntelCpuBFloat16NaNTest,
    ::testing::Values(IntelCpuBFloat16NaNTestCase{"positive_signaling_low_payload", 0x7f800001U, 0x7fc0},
                      IntelCpuBFloat16NaNTestCase{"negative_signaling_low_payload", 0xff800001U, 0xffc0},
                      IntelCpuBFloat16NaNTestCase{"positive_nan_near_sign_boundary", 0x7fff8000U, 0x7fff},
                      IntelCpuBFloat16NaNTestCase{"negative_nan_near_sign_boundary", 0xffff8000U, 0xffff},
                      IntelCpuBFloat16NaNTestCase{"positive_quiet_payload", 0x7fc12345U, 0x7fc1},
                      IntelCpuBFloat16NaNTestCase{"negative_quiet_payload", 0xffc12345U, 0xffc1}),
    [](const testing::TestParamInfo<IntelCpuBFloat16NaNTestCase>& info) {
        return info.param.name;
    });

TEST_P(IntelCpuBFloat16NaNTest, converts_to_quiet_bfloat16_nan) {
    const auto& test_case = GetParam();
    const auto input = ov::intel_cpu::bit_cast<float>(test_case.input);
    const auto actual = bfloat16_t(input).to_bits();

    EXPECT_EQ(actual, test_case.expected);
    EXPECT_EQ(actual & 0x7f80U, 0x7f80U);
    EXPECT_NE(actual & 0x007fU, 0U);
    EXPECT_NE(actual & 0x0040U, 0U);
}

}  // namespace ov::intel_cpu::test
