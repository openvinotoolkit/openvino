// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float4_e2m1.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <cmath>

#include "common_test_utils/float_util.hpp"

namespace ov {
namespace test {

template <class TContainer>
std::vector<std::tuple<int, typename TContainer::value_type>> enumerate(const TContainer& values) {
    std::vector<std::tuple<int, typename TContainer::value_type>> enum_values;
    int i = 0;
    for (const auto& v : values) {
        enum_values.emplace_back(i, v);
        ++i;
    }
    return enum_values;
}

TEST(F4E2M1Test, f32_inf) {
    const auto f4 = ov::float4_e2m1(std::numeric_limits<float>::infinity());
    // f4 is max as there is no infinity
    EXPECT_EQ(f4.to_bits(), 0b0111);
}

TEST(F4E2M1Test, f32_minus_inf) {
    const auto f4 = ov::float4_e2m1(-std::numeric_limits<float>::infinity());
    // f4 is max as there is no infinity
    EXPECT_EQ(f4.to_bits(), 0b1111);
}

TEST(F4E2M1Test, f32_gt_zero_round_to_f4_zero) {
    const auto f4 = ov::float4_e2m1(0.218749985098838806152f);

    EXPECT_EQ(f4.to_bits(), 0b0000);
}

TEST(F4E2M1Test, f32_gt_zero_round_to_f4_lowest_subnormal) {
    const auto f4 = ov::float4_e2m1(0.21875f);

    EXPECT_EQ(f4.to_bits(), 0b0001);
}

TEST(F4E2M1Test, f32_normal_fractional_rounding) {
    const auto f4 = ov::float4_e2m1(1.75f);

    EXPECT_EQ(f4.to_bits(), 0b0100);
}

TEST(F4E2M1Test, f32_normal_negative_fractional_rounding) {
    const auto f4 = ov::float4_e2m1(-2.1f);

    EXPECT_EQ(f4.to_bits(), 0b1100);
}

TEST(F4E2M1Test, f32_ge_f4_max_within_round_to_max) {
    const auto f4 = ov::float4_e2m1(6.1f);

    EXPECT_EQ(f4.to_bits(), 0b0111);
}

TEST(F4E2M1Test, f32_ge_f8_max_not_within_round_to_max) {
    const auto f4 = ov::float4_e2m1(7.0f);

    EXPECT_EQ(f4.to_bits(), 0b0111);
}

TEST(F4E2M1Test, f32_le_f8_lowest_within_round_to_lowest) {
    const auto f4 = ov::float4_e2m1(-6.5f);

    EXPECT_EQ(f4.to_bits(), 0b1111);
}

TEST(F4E2M1Test, f32_le_f8_lowest_not_within_round_to_lowest) {
    const auto f4 = ov::float4_e2m1(-7.0f);

    EXPECT_EQ(f4.to_bits(), 0b1111);
}

TEST(F4E2M1Test, stream_operator) {
    std::stringstream s;
    s << ov::float4_e2m1(-1.5f);

    EXPECT_EQ(s.str(), "-1.5");
}

TEST(F4E2M1Test, to_string) {
    const auto f4 = ov::float4_e2m1::from_bits(0b00000110);

    EXPECT_EQ(std::to_string(f4), "4.000000");
}

// clang-format off
const auto exp_floats = std::vector<float>{
    0.0f,       0.5f,
    1.0f,       1.5f,
    2.0f,       3.0f,
    4.0f,       6.0f,
    -0.0f,      -0.5f,
    -1.0f,      -1.5f,
    -2.0f,      -3.0f,
    -4.0f,      -6.0f};
// clang-format on

using f4m2e1_params = std::tuple<int, float>;
class F4E2M1PTest : public testing::TestWithParam<f4m2e1_params> {};

INSTANTIATE_TEST_SUITE_P(convert,
                         F4E2M1PTest,
                         testing::ValuesIn(enumerate(exp_floats)),
                         testing::PrintToStringParamName());

TEST_P(F4E2M1PTest, f4_bits_to_f32) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<1>(params);
    const auto f4 = ov::float4_e2m1::from_bits(std::get<0>(params));

    EXPECT_EQ(static_cast<float>(f4), exp_value);
}

TEST_P(F4E2M1PTest, f32_to_f4_bits) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<0>(params);
    const auto& value = std::get<1>(params);
    const auto f4 = ov::float4_e2m1(value);

    EXPECT_EQ(f4.to_bits(), exp_value);
}

TEST(F4E2M1Test, f48e2m1_num_limits_exp) {
    const auto min_exp = std::numeric_limits<ov::float4_e2m1>::min_exponent;
    const auto min_exp10 = std::numeric_limits<ov::float4_e2m1>::min_exponent10;
    const auto max_exp = std::numeric_limits<ov::float4_e2m1>::max_exponent;
    const auto max_exp10 = std::numeric_limits<ov::float4_e2m1>::max_exponent10;

    EXPECT_EQ(min_exp, 1);
    EXPECT_EQ(min_exp10, 0);
    EXPECT_EQ(max_exp, 3);
    EXPECT_EQ(max_exp10, 0);
}

TEST(F4E2M1Test, min_normalized) {
    const auto f4 = ov::float4_e2m1(0.8f);

    EXPECT_EQ(f4.to_bits(), 0b0010);
    EXPECT_EQ(f4.to_bits(), std::numeric_limits<ov::float4_e2m1>::min().to_bits());
}

TEST(F4E2M1Test, max_normalized) {
    const auto f4 = ov::float4_e2m1(6.0f);

    EXPECT_EQ(f4.to_bits(), 0b0111);
    EXPECT_EQ(f4.to_bits(), std::numeric_limits<ov::float4_e2m1>::max().to_bits());
}

TEST(F4E2M1Test, lowest_normalized) {
    const auto f4 = ov::float4_e2m1(-6.0f);

    EXPECT_EQ(f4.to_bits(), 0b1111);
    EXPECT_EQ(f4.to_bits(), std::numeric_limits<ov::float4_e2m1>::lowest().to_bits());
}

TEST(F4E2M1Test, denorm_min) {
    const auto f4 = ov::float4_e2m1(0.5f);

    EXPECT_EQ(f4.to_bits(), 0b0001);
    EXPECT_EQ(f4.to_bits(), std::numeric_limits<ov::float4_e2m1>::denorm_min().to_bits());
}

TEST(F4E2M1Test, num_limits_is_specialized) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::is_specialized;
    EXPECT_TRUE(val);
}

TEST(F4E2M1Test, num_limits_is_signed) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::is_signed;
    EXPECT_TRUE(val);
}

TEST(F4E2M1Test, num_limits_is_integer) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::is_integer;
    EXPECT_FALSE(val);
}

TEST(F4E2M1Test, num_limits_is_exact) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::is_exact;
    EXPECT_FALSE(val);
}

TEST(F4E2M1Test, num_limits_radix) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::radix;
    EXPECT_EQ(val, 2);
}

TEST(F4E2M1Test, num_limits_digits) {
    const auto val = std::numeric_limits<ov::float4_e2m1>::digits;
    EXPECT_EQ(val, 2);
}

TEST(F4E2M1Test, num_limits_digits10) {
    const auto f4_dig = std::numeric_limits<ov::float4_e2m1>::digits;
    const auto f4_dig10 = std::numeric_limits<ov::float4_e2m1>::digits10;

    EXPECT_EQ(f4_dig10, static_cast<int>((f4_dig - 1) * std::log10(2)));
    EXPECT_EQ(f4_dig10, 0);
}

TEST(F4E2M1Test, num_limits_epsilon) {
    const auto f4_1 = ov::float4_e2m1(1.f);
    const auto f4_1_bits = f4_1.to_bits();
    const auto f4_1_next_bits = f4_1_bits + 1u;

    const auto f4_eps = ov::float4_e2m1::from_bits(f4_1_next_bits - f4_1_bits);

    EXPECT_EQ(f4_eps, std::numeric_limits<ov::float4_e2m1>::epsilon());
    EXPECT_EQ(f4_eps.to_bits(), std::numeric_limits<ov::float4_e2m1>::epsilon().to_bits());
}

TEST(F4E2M1Test, num_limits_round_error) {
    const auto f4 = ov::float4_e2m1(0.5f);

    EXPECT_EQ(f4, std::numeric_limits<ov::float4_e2m1>::round_error());
    EXPECT_EQ(f4.to_bits(), std::numeric_limits<ov::float4_e2m1>::round_error().to_bits());
}

TEST(F4E2M1Test, quiet_nan) {
    const auto has_quiet_nan = std::numeric_limits<ov::float4_e2m1>::has_quiet_NaN;
    EXPECT_FALSE(has_quiet_nan);
    EXPECT_EQ(std::numeric_limits<ov::float4_e2m1>::quiet_NaN().to_bits(), 0);
}

TEST(F4E2M1Test, f32_quiet_nan) {
    const auto f4 = ov::float4_e2m1(std::numeric_limits<float>::quiet_NaN());

    EXPECT_FALSE(std::isnan(f4));
    EXPECT_EQ(f4.to_bits(), 0b0111);
    EXPECT_EQ(0, std::numeric_limits<ov::float4_e2m1>::quiet_NaN().to_bits());
}

TEST(F4E2M1Test, f32_sig_nan) {
    const auto f4 = ov::float4_e2m1(std::numeric_limits<float>::signaling_NaN());

    const auto has_sig_nan = std::numeric_limits<ov::float4_e2m1>::has_signaling_NaN;
    EXPECT_FALSE(has_sig_nan);
    EXPECT_FALSE(std::isnan(f4));
    EXPECT_EQ(f4.to_bits(), 0b0111);
    EXPECT_EQ(0, std::numeric_limits<ov::float4_e2m1>::signaling_NaN().to_bits());
}
}  // namespace test
}  // namespace ov
