// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e4m3.hpp"

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

TEST(F8E4M3Test, f32_inf) {
    const auto f8 = ov::float8_e4m3(std::numeric_limits<float>::infinity());
    // f8 is NaN as there is no infinity
    EXPECT_EQ(f8.to_bits(), 0x7f);
}

TEST(F8E4M3Test, f32_minus_inf) {
    const auto f8 = ov::float8_e4m3(-std::numeric_limits<float>::infinity());
    // f8 is NaN as there is no infinity
    EXPECT_EQ(f8.to_bits(), 0xff);
}

TEST(F8E4M3Test, f8e4m3_num_limits_is_specialized) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::is_specialized;
    EXPECT_TRUE(val);
}

TEST(F8E4M3Test, f8e4m3_num_limits_is_signed) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::is_signed;
    EXPECT_TRUE(val);
}

TEST(F8E4M3Test, f8e4m3_num_limits_is_integer) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::is_integer;
    EXPECT_FALSE(val);
}

TEST(F8E4M3Test, f8e4m3_num_limits_is_exact) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::is_exact;
    EXPECT_FALSE(val);
}

TEST(F8E4M3Test, f8e4m3_num_limits_radix) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::radix;
    EXPECT_EQ(val, 2);
}

TEST(F8E4M3Test, f8e4m3_num_limits_digits) {
    const auto val = std::numeric_limits<ov::float8_e4m3>::digits;
    EXPECT_EQ(val, 4);
}

TEST(F8E4M3Test, f8e4m3_num_limits_digits10) {
    const auto f8_dig = std::numeric_limits<ov::float8_e4m3>::digits;
    const auto f8_dig10 = std::numeric_limits<ov::float8_e4m3>::digits10;

    EXPECT_EQ(f8_dig10, static_cast<int>((f8_dig - 1) * std::log10(2)));
    EXPECT_EQ(f8_dig10, 0);
}

TEST(F8E4M3Test, f8e4m3_num_limits_epsilon) {
    const auto f8_1 = ov::float8_e4m3(1.f);
    const auto f8_1_bits = f8_1.to_bits();
    const auto f8_1_next_bits = f8_1_bits + 1u;

    const auto f8_eps = ov::float8_e4m3::from_bits(f8_1_next_bits - f8_1_bits);

    EXPECT_EQ(f8_eps, std::numeric_limits<ov::float8_e4m3>::epsilon());
    EXPECT_EQ(f8_eps.to_bits(), std::numeric_limits<ov::float8_e4m3>::epsilon().to_bits());
}

TEST(F8E4M3Test, f8e4m3_num_limits_round_error) {
    const auto f8 = ov::float8_e4m3(0.5f);

    EXPECT_EQ(f8, std::numeric_limits<ov::float8_e4m3>::round_error());
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::round_error().to_bits());
}

TEST(F8E4M3Test, f8e4m3_quiet_nan) {
    const auto has_quiet_nan = std::numeric_limits<ov::float8_e4m3>::has_quiet_NaN;
    EXPECT_TRUE(has_quiet_nan);
    EXPECT_EQ(std::numeric_limits<ov::float8_e4m3>::quiet_NaN().to_bits(), 0b01111111);
}

TEST(F8E4M3Test, f32_quiet_nan) {
    const auto f8 = ov::float8_e4m3(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111111);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::quiet_NaN().to_bits());
}

TEST(F8E4M3Test, f32_sig_nan) {
    const auto f8 = ov::float8_e4m3(std::numeric_limits<float>::signaling_NaN());

    const auto has_sig_nan = std::numeric_limits<ov::float8_e4m3>::has_signaling_NaN;
    EXPECT_FALSE(has_sig_nan);
    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111111);
    EXPECT_EQ(0, std::numeric_limits<ov::float8_e4m3>::signaling_NaN().to_bits());
}

TEST(F8E4M3Test, f8e4m3_min_normalized) {
    const auto f8 = ov::float8_e4m3(0.015625f);

    EXPECT_EQ(f8.to_bits(), 0b00001000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::min().to_bits());
}

TEST(F8E4M3Test, f8e4m3_max_normalized) {
    const auto f8 = ov::float8_e4m3(448.0f);

    EXPECT_EQ(f8.to_bits(), 0b01111110);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::max().to_bits());
}

TEST(F8E4M3Test, f8e4m3_lowest_normalized) {
    const auto f8 = ov::float8_e4m3(-448.0f);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::lowest().to_bits());
}

TEST(F8E4M3Test, f8e4m3_denorm_min) {
    const auto f8 = ov::float8_e4m3(0.001953125f);

    EXPECT_EQ(f8.to_bits(), 0b00000001);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e4m3>::denorm_min().to_bits());
}

TEST(F8E4M3Test, f8e4m3_num_limits_exp) {
    const auto min_exp = std::numeric_limits<ov::float8_e4m3>::min_exponent;
    const auto min_exp10 = std::numeric_limits<ov::float8_e4m3>::min_exponent10;
    const auto max_exp = std::numeric_limits<ov::float8_e4m3>::max_exponent;
    const auto max_exp10 = std::numeric_limits<ov::float8_e4m3>::max_exponent10;

    EXPECT_EQ(min_exp, -5);
    EXPECT_EQ(min_exp10, -2);
    EXPECT_EQ(max_exp, 10);
    EXPECT_EQ(max_exp10, 2);
}

TEST(F8E4M3Test, f32_gt_zero_le_f8_half_lowest_subnormal) {
    const auto f8 = ov::float8_e4m3(0.0009765625f);

    EXPECT_EQ(f8.to_bits(), 0x00);
}

TEST(F8E4M3Test, f32_in_f16_format_le_zero_gt_f8_half_lowest_subnormal) {
    const auto f8 = ov::float8_e4m3(0.00097656273283064365387f);

    EXPECT_EQ(f8.to_bits(), 0x00);
}

TEST(F8E4M3Test, f32_in_f16_format_gt_zero_gt_f8_half_lowest_subnormal) {
    const auto f8 = ov::float8_e4m3(0.00097751617431640625f);

    EXPECT_EQ(f8.to_bits(), 0x01);
}

TEST(F8E4M3Test, f32_normal_fractional_rounding) {
    const auto f8 = ov::float8_e4m3(0.129f);

    // Rounded to 0.140625f -> 0x21
    EXPECT_EQ(f8.to_bits(), 0x20);
}

TEST(F8E4M3Test, f32_normal_negative_fractional_rounding) {
    const auto f8 = ov::float8_e4m3(-0.281f);

    // Rounded to -0.28125f -> 0x21
    EXPECT_EQ(f8.to_bits(), 0xa9);
}

TEST(F8E4M3Test, f32_ge_f8_max_within_round_to_max) {
    const auto f8 = ov::float8_e4m3(460.0f);

    // Rounded to 448.0f -> 0x7e
    EXPECT_EQ(f8.to_bits(), 0x7e);
}

TEST(F8E4M3Test, f32_ge_f8_max_not_within_round_to_max) {
    const auto f8 = ov::float8_e4m3(560.0f);

    // f8 has no such value (NaN)
    EXPECT_EQ(f8.to_bits(), 0x7f);
}

TEST(F8E4M3Test, f32_le_f8_lowest_within_round_to_lowest) {
    const auto f8 = ov::float8_e4m3(-460.0f);

    // Rounded to -448.0f -> 0xfe
    EXPECT_EQ(f8.to_bits(), 0xfe);
}

TEST(F8E4M3Test, f32_le_f8_lowest_not_within_round_to_lowest) {
    const auto f8 = ov::float8_e4m3(-760.0f);

    // f8 has no such value (NaN)
    EXPECT_EQ(f8.to_bits(), 0xff);
}

TEST(F8E4M3Test, stream_operator) {
    std::stringstream s;
    s << ov::float8_e4m3(2.5f);

    EXPECT_EQ(s.str(), "2.5");
}

TEST(F8E4M3Test, to_string) {
    const auto f8 = ov::float8_e4m3::from_bits(0b00111010);

    EXPECT_EQ(std::to_string(f8), "1.250000");
}
constexpr auto f32_qnan = std::numeric_limits<float>::quiet_NaN();

const auto exp_floats = std::vector<float>{
    0.0f,       0.001953125f,  0.00390625f,  0.005859375f,  0.0078125f,  0.009765625f,  0.01171875f,  0.013671875f,
    0.015625f,  0.017578125f,  0.01953125f,  0.021484375f,  0.0234375f,  0.025390625f,  0.02734375f,  0.029296875f,
    0.03125f,   0.03515625f,   0.0390625f,   0.04296875f,   0.046875f,   0.05078125f,   0.0546875f,   0.05859375f,
    0.0625f,    0.0703125f,    0.078125f,    0.0859375f,    0.09375f,    0.1015625f,    0.109375f,    0.1171875f,
    0.125f,     0.140625f,     0.15625f,     0.171875f,     0.1875f,     0.203125f,     0.21875f,     0.234375f,
    0.25f,      0.28125f,      0.3125f,      0.34375f,      0.375f,      0.40625f,      0.4375f,      0.46875f,
    0.5f,       0.5625f,       0.625f,       0.6875f,       0.75f,       0.8125f,       0.875f,       0.9375f,
    1.0f,       1.125f,        1.25f,        1.375f,        1.5f,        1.625f,        1.75f,        1.875f,
    2.0f,       2.25f,         2.5f,         2.75f,         3.0f,        3.25f,         3.5f,         3.75f,
    4.0f,       4.5f,          5.0f,         5.5f,          6.0f,        6.5f,          7.0f,         7.5f,
    8.0f,       9.0f,          10.0f,        11.0f,         12.0f,       13.0f,         14.0f,        15.0f,
    16.0f,      18.0f,         20.0f,        22.0f,         24.0f,       26.0f,         28.0f,        30.0f,
    32.0f,      36.0f,         40.0f,        44.0f,         48.0f,       52.0f,         56.0f,        60.0f,
    64.0f,      72.0f,         80.0f,        88.0f,         96.0f,       104.0f,        112.0f,       120.0f,
    128.0f,     144.0f,        160.0f,       176.0f,        192.0f,      208.0f,        224.0f,       240.0f,
    256.0f,     288.0f,        320.0f,       352.0f,        384.0f,      416.0f,        448.0f,       f32_qnan,
    -0.0f,      -0.001953125f, -0.00390625f, -0.005859375f, -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f, -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f,  -0.03515625f,  -0.0390625f,  -0.04296875f,  -0.046875f,  -0.05078125f,  -0.0546875f,  -0.05859375f,
    -0.0625f,   -0.0703125f,   -0.078125f,   -0.0859375f,   -0.09375f,   -0.1015625f,   -0.109375f,   -0.1171875f,
    -0.125f,    -0.140625f,    -0.15625f,    -0.171875f,    -0.1875f,    -0.203125f,    -0.21875f,    -0.234375f,
    -0.25f,     -0.28125f,     -0.3125f,     -0.34375f,     -0.375f,     -0.40625f,     -0.4375f,     -0.46875f,
    -0.5f,      -0.5625f,      -0.625f,      -0.6875f,      -0.75f,      -0.8125f,      -0.875f,      -0.9375f,
    -1.0f,      -1.125f,       -1.25f,       -1.375f,       -1.5f,       -1.625f,       -1.75f,       -1.875f,
    -2.0f,      -2.25f,        -2.5f,        -2.75f,        -3.0f,       -3.25f,        -3.5f,        -3.75f,
    -4.0f,      -4.5f,         -5.0f,        -5.5f,         -6.0f,       -6.5f,         -7.0f,        -7.5f,
    -8.0f,      -9.0f,         -10.0f,       -11.0f,        -12.0f,      -13.0f,        -14.0f,       -15.0f,
    -16.0f,     -18.0f,        -20.0f,       -22.0f,        -24.0f,      -26.0f,        -28.0f,       -30.0f,
    -32.0f,     -36.0f,        -40.0f,       -44.0f,        -48.0f,      -52.0f,        -56.0f,       -60.0f,
    -64.0f,     -72.0f,        -80.0f,       -88.0f,        -96.0f,      -104.0f,       -112.0f,      -120.0f,
    -128.0f,    -144.0f,       -160.0f,      -176.0f,       -192.0f,     -208.0f,       -224.0f,      -240.0f,
    -256.0f,    -288.0f,       -320.0f,      -352.0f,       -384.0f,     -416.0f,       -448.0f,      -f32_qnan};

using f8m4e3_params = std::tuple<int, float>;
class F8E4M3PTest : public testing::TestWithParam<f8m4e3_params> {};

INSTANTIATE_TEST_SUITE_P(convert,
                         F8E4M3PTest,
                         testing::ValuesIn(enumerate(exp_floats)),
                         testing::PrintToStringParamName());

TEST_P(F8E4M3PTest, f8_bits_to_f32) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<1>(params);
    const auto f8 = ov::float8_e4m3::from_bits(std::get<0>(params));

    if (std::isnan(exp_value)) {
        EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
    } else {
        EXPECT_EQ(static_cast<float>(f8), exp_value);
    }
}

TEST_P(F8E4M3PTest, f32_to_f8_bits) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<0>(params);
    const auto& value = std::get<1>(params);
    const auto f8 = ov::float8_e4m3(value);

    EXPECT_EQ(f8.to_bits(), exp_value);
}
}  // namespace test
}  // namespace ov
