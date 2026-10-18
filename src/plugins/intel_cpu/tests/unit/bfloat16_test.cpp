// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/bfloat16.hpp"

#include <gtest/gtest.h>

#include "cpp/bit_cast.hpp"

namespace ov::intel_cpu::test {

TEST(bfloat16, converts_nans_to_quiet_bfloat16_nans) {
    struct TestCase {
        uint32_t input;
        uint16_t expected;
    };
    const TestCase cases[] = {
        {0x7f800001U, 0x7fc0},  // positive signaling NaN, low payload only
        {0xff800001U, 0xffc0},  // negative signaling NaN, low payload only
        {0x7fff8000U, 0x7fff},  // positive NaN mantissa near sign boundary
        {0xffff8000U, 0xffff},  // negative NaN mantissa near sign boundary
        {0x7fc12345U, 0x7fc1},  // positive quiet NaN
        {0xffc12345U, 0xffc1},  // negative quiet NaN
    };

    for (const auto& test_case : cases) {
        const auto input = ov::intel_cpu::bit_cast<float>(test_case.input);
        const auto actual = bfloat16_t(input).to_bits();
        EXPECT_EQ(actual, test_case.expected);
        EXPECT_EQ(actual & 0x7f80, 0x7f80);
        EXPECT_NE(actual & 0x007f, 0);
    }
}

}  // namespace ov::intel_cpu::test
