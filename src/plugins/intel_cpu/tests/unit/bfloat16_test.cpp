// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/bfloat16.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <string>

#include "utils/cpp/bit_cast.hpp"

namespace ov::intel_cpu::test {

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
