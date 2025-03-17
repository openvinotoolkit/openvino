// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e5m2.hpp"

#include <gtest/gtest.h>

#include <climits>

#include "common_test_utils/float_util.hpp"

namespace ov {
namespace test {

TEST(F8E5M2Test, stream_operator) {
    std::stringstream s;
    s << ov::float8_e5m2(2.5f);

    EXPECT_EQ(s.str(), "2.5");
}

TEST(F8E5M2Test, to_string) {
    const auto f8 = ov::float8_e5m2::from_bits(0b00111010);

    EXPECT_EQ(std::to_string(f8), "0.750000");
}

TEST(F8E5M2Test, f32_inf) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<float>::infinity());

    EXPECT_EQ(f8.to_bits(), 0b01111100);
    EXPECT_EQ(f8, std::numeric_limits<float8_e5m2>::infinity());
}

TEST(F8E5M2Test, f32_minus_inf) {
    const auto f8 = ov::float8_e5m2(-std::numeric_limits<float>::infinity());

    EXPECT_EQ(f8.to_bits(), 0b11111100);
    EXPECT_EQ(f8, -std::numeric_limits<float8_e5m2>::infinity());
}

TEST(F8E5M2Test, f8e5m2_num_limits_is_specialized) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::is_specialized;
    EXPECT_TRUE(val);
}

TEST(F8E5M2Test, f8e5m2_num_limits_is_signed) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::is_signed;
    EXPECT_TRUE(val);
}

TEST(F8E5M2Test, f8e5m2_num_limits_is_integer) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::is_integer;
    EXPECT_FALSE(val);
}

TEST(F8E5M2Test, f8e5m2_num_limits_is_exact) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::is_exact;
    EXPECT_FALSE(val);
}

TEST(F8E5M2Test, f8e5m2_num_limits_radix) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::radix;
    EXPECT_EQ(val, 2);
}

TEST(F8E5M2Test, f8e5m2_num_limits_digits) {
    const auto val = std::numeric_limits<ov::float8_e5m2>::digits;
    EXPECT_EQ(val, 3);
}

TEST(F8E5M2Test, f8e5m2_num_limits_digits10) {
    const auto f8_dig = std::numeric_limits<ov::float8_e5m2>::digits;
    const auto f8_dig10 = std::numeric_limits<ov::float8_e5m2>::digits10;

    EXPECT_EQ(f8_dig10, static_cast<int>((f8_dig - 1) * std::log10(2)));
    EXPECT_EQ(f8_dig10, 0);
}

TEST(F8E5M2Test, f8e5m2_num_limits_epsilon) {
    const auto f8_1 = ov::float8_e5m2(1.f);
    const auto f8_1_bits = f8_1.to_bits();
    const auto f8_1_next_bits = f8_1_bits + 1u;

    const auto f8_eps = ov::float8_e5m2::from_bits(f8_1_next_bits - f8_1_bits);

    EXPECT_EQ(f8_eps, std::numeric_limits<ov::float8_e5m2>::epsilon());
    EXPECT_EQ(f8_eps.to_bits(), std::numeric_limits<ov::float8_e5m2>::epsilon().to_bits());
}

TEST(F8E5M2Test, f8e5m2_num_limits_round_error) {
    const auto f8 = ov::float8_e5m2(0.5f);

    EXPECT_EQ(f8, std::numeric_limits<ov::float8_e5m2>::round_error());
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e5m2>::round_error().to_bits());
}

TEST(F8E5M2Test, f8e5m2_quiet_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<ov::float8_e5m2>::quiet_NaN());

    const auto has_quiet_nan = std::numeric_limits<ov::float8_e5m2>::has_quiet_NaN;
    EXPECT_TRUE(has_quiet_nan);
    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111111);
    EXPECT_EQ(std::numeric_limits<ov::float8_e5m2>::quiet_NaN().to_bits(), 0b01111111);
}

TEST(F8E5M2Test, f8e5m2_sig_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<ov::float8_e5m2>::signaling_NaN());

    const auto has_sig_nan = std::numeric_limits<ov::float8_e5m2>::has_signaling_NaN;
    EXPECT_TRUE(has_sig_nan);
    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111101);
    EXPECT_EQ(std::numeric_limits<ov::float8_e5m2>::signaling_NaN().to_bits(), 0b01111101);
}

TEST(F8E5M2Test, f16_quiet_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<ov::float16>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111111);
    EXPECT_EQ(std::numeric_limits<ov::float8_e5m2>::quiet_NaN().to_bits(), 0b01111111);
}

TEST(F8E5M2Test, f16_sig_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<ov::float16>::signaling_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b01111101);
    EXPECT_EQ(std::numeric_limits<ov::float8_e5m2>::signaling_NaN().to_bits(), 0b01111101);
}

TEST(F8E5M2Test, f32_quiet_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    // The last bit is zero because of f32 to f16 conversion
    EXPECT_EQ(f8.to_bits(), 0b01111110);
}

TEST(F8E5M2Test, f32_sig_nan) {
    const auto f8 = ov::float8_e5m2(std::numeric_limits<float>::signaling_NaN());
    const auto f8_bits = f8.to_bits();

    const auto allowed_nan = (f8_bits == 0b01111101) || (f8_bits == 0b01111110);
    EXPECT_TRUE(allowed_nan);

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(std::numeric_limits<ov::float8_e5m2>::signaling_NaN().to_bits(), 0b01111101);
}

TEST(F8E5M2Test, f8e5m2_min_normalized) {
    const auto f8 = ov::float8_e5m2(0.00006103515625f);

    EXPECT_EQ(f8.to_bits(), 0b00000100);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e5m2>::min().to_bits());
}

TEST(F8E5M2Test, f8e5m2_max_normalized) {
    const auto f8 = ov::float8_e5m2(57344.0f);

    EXPECT_EQ(f8.to_bits(), 0b01111011);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e5m2>::max().to_bits());
}

TEST(F8E5M2Test, f8e5m2_lowest_normalized) {
    const auto f8 = ov::float8_e5m2(-57344.0f);

    EXPECT_EQ(f8.to_bits(), 0b11111011);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e5m2>::lowest().to_bits());
}

TEST(F8E5M2Test, f8e5m2_denorm_min) {
    const auto f8 = ov::float8_e5m2(0.0000152587890625f);

    EXPECT_EQ(f8.to_bits(), 0b00000001);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e5m2>::denorm_min().to_bits());
}

TEST(F8E5M2Test, f8e5m2_num_limits_exp) {
    const auto min_exp = std::numeric_limits<ov::float8_e5m2>::min_exponent;
    const auto min_exp10 = std::numeric_limits<ov::float8_e5m2>::min_exponent10;
    const auto max_exp = std::numeric_limits<ov::float8_e5m2>::max_exponent;
    const auto max_exp10 = std::numeric_limits<ov::float8_e5m2>::max_exponent10;

    EXPECT_EQ(min_exp, -13);
    EXPECT_EQ(min_exp10, -4);
    EXPECT_EQ(max_exp, 16);
    EXPECT_EQ(max_exp10, 4);
}

TEST(F8E5M2Test, f32_as_f16_inf_gt_f8_max_round_to_inf) {
    const auto f8 = ov::float8_e5m2(65520.0f);

    EXPECT_EQ(f8.to_bits(), 0b01111100);
}

TEST(F8E5M2Test, f32_gt_f16_max_gt_f8_max_round_to_inf) {
    const auto f8 = ov::float8_e5m2(65519.9f);

    EXPECT_EQ(f8.to_bits(), 0b01111100);
}

TEST(F8E5M2Test, f32_as_f16_minus_inf_lt_f8_lowest_round_to_minus_inf) {
    const auto f8 = ov::float8_e5m2(-65520.0f);

    EXPECT_EQ(f8.to_bits(), 0b11111100);
}

TEST(F8E5M2Test, f32_lt_f16_lowest_lt_f8_lowest_round_to_minus_inf) {
    const auto f8 = ov::float8_e5m2(-65519.9f);

    EXPECT_EQ(f8.to_bits(), 0b11111100);
}

template <class TContainer>
std::vector<std::tuple<uint8_t, typename TContainer::value_type>> enumerate(const TContainer& values) {
    std::vector<std::tuple<uint8_t, typename TContainer::value_type>> enum_values;
    uint8_t i = 0;
    for (const auto& v : values) {
        enum_values.emplace_back(i, v);
        ++i;
    }
    return enum_values;
}

constexpr auto f32_qnan = std::numeric_limits<float>::quiet_NaN();
constexpr auto f32_signan = std::numeric_limits<float>::signaling_NaN();
constexpr auto f32_inf = std::numeric_limits<float>::infinity();

// clang-format off
const auto exp_floats = std::vector<float>{
    0.0f,               1.52587890625e-05f,     3.0517578125e-05f,      4.57763671875e-05f,
    6.103515625e-05f,   7.62939453125e-05f,     9.1552734375e-05f,      0.0001068115234375f,
    0.0001220703125f,   0.000152587890625f,     0.00018310546875f,      0.000213623046875f,
    0.000244140625f,    0.00030517578125f,      0.0003662109375f,       0.00042724609375f,
    0.00048828125f,     0.0006103515625f,       0.000732421875f,        0.0008544921875f,
    0.0009765625f,      0.001220703125f,        0.00146484375f,         0.001708984375f,
    0.001953125f,       0.00244140625f,         0.0029296875f,          0.00341796875f,
    0.00390625f,        0.0048828125f,          0.005859375f,           0.0068359375f,
    0.0078125f,         0.009765625f,           0.01171875f,            0.013671875f,
    0.015625f,          0.01953125f,            0.0234375f,             0.02734375f,
    0.03125f,           0.0390625f,             0.046875f,              0.0546875f,
    0.0625f,            0.078125f,              0.09375f,               0.109375f,
    0.125f,             0.15625f,               0.1875f,                0.21875f,
    0.25f,              0.3125f,                0.375f,                 0.4375f,
    0.5f,               0.625f,                 0.75f,                  0.875f,
    1.0f,               1.25f,                  1.5f,                   1.75f,
    2.0f,               2.5f,                   3.0f,                   3.5f,
    4.0f,               5.0f,                   6.0f,                   7.0f,
    8.0f,               10.0f,                  12.0f,                  14.0f,
    16.0f,              20.0f,                  24.0f,                  28.0f,
    32.0f,              40.0f,                  48.0f,                  56.0f,
    64.0f,              80.0f,                  96.0f,                  112.0f,
    128.0f,             160.0f,                 192.0f,                 224.0f,
    256.0f,             320.0f,                 384.0f,                 448.0f,
    512.0f,             640.0f,                 768.0f,                 896.0f,
    1024.0f,            1280.0f,                1536.0f,                1792.0f,
    2048.0f,            2560.0f,                3072.0f,                3584.0f,
    4096.0f,            5120.0f,                6144.0f,                7168.0f,
    8192.0f,            10240.0f,               12288.0f,               14336.0f,
    16384.0f,           20480.0f,               24576.0f,               28672.0f,
    32768.0f,           40960.0f,               49152.0f,               57344.0f,
    f32_inf,            f32_signan,             f32_qnan,               f32_qnan,
    -0.0f,              -1.52587890625e-05f,    -3.0517578125e-05f,     -4.57763671875e-05f,
    -6.103515625e-05f,  -7.62939453125e-05f,    -9.1552734375e-05f,     -0.0001068115234375f,
    -0.0001220703125f,  -0.000152587890625f,    -0.00018310546875f,     -0.000213623046875f,
    -0.000244140625f,   -0.00030517578125f,     -0.0003662109375f,      -0.00042724609375f,
    -0.00048828125f,    -0.0006103515625f,      -0.000732421875f,       -0.0008544921875f,
    -0.0009765625f,     -0.001220703125f,       -0.00146484375f,        -0.001708984375f,
    -0.001953125f,      -0.00244140625f,        -0.0029296875f,         -0.00341796875f,
    -0.00390625f,       -0.0048828125f,         -0.005859375f,          -0.0068359375f,
    -0.0078125f,        -0.009765625f,          -0.01171875f,           -0.013671875f,
    -0.015625f,         -0.01953125f,           -0.0234375f,            -0.02734375f,
    -0.03125f,          -0.0390625f,            -0.046875f,             -0.0546875f,
    -0.0625f,           -0.078125f,             -0.09375f,              -0.109375f,
    -0.125f,            -0.15625f,              -0.1875f,               -0.21875f,
    -0.25f,             -0.3125f,               -0.375f,                -0.4375f,
    -0.5f,              -0.625f,                -0.75f,                 -0.875f,
    -1.0f,              -1.25f,                 -1.5f,                  -1.75f,
    -2.0f,              -2.5f,                  -3.0f,                  -3.5f,
    -4.0f,              -5.0f,                  -6.0f,                  -7.0f,
    -8.0f,              -10.0f,                 -12.0f,                 -14.0f,
    -16.0f,             -20.0f,                 -24.0f,                 -28.0f,
    -32.0f,             -40.0f,                 -48.0f,                 -56.0f,
    -64.0f,             -80.0f,                 -96.0f,                 -112.0f,
    -128.0f,            -160.0f,                -192.0f,                -224.0f,
    -256.0f,            -320.0f,                -384.0f,                -448.0f,
    -512.0f,            -640.0f,                -768.0f,                -896.0f,
    -1024.0f,           -1280.0f,               -1536.0f,               -1792.0f,
    -2048.0f,           -2560.0f,               -3072.0f,               -3584.0f,
    -4096.0f,           -5120.0f,               -6144.0f,               -7168.0f,
    -8192.0f,           -10240.0f,              -12288.0f,              -14336.0f,
    -16384.0f,          -20480.0f,              -24576.0f,              -28672.0f,
    -32768.0f,          -40960.0f,              -49152.0f,              -57344.0f,
    -f32_inf,           -f32_signan,            -f32_qnan,              -f32_qnan};
// clang-format on

using f8m5e2_params = std::tuple<uint8_t, float>;
class F8E5M2PTest : public testing::TestWithParam<f8m5e2_params> {};

INSTANTIATE_TEST_SUITE_P(convert,
                         F8E5M2PTest,
                         testing::ValuesIn(enumerate(exp_floats)),
                         testing::PrintToStringParamName());

TEST_P(F8E5M2PTest, f8_bits_to_f32) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<1>(params);
    const auto f8 = ov::float8_e5m2::from_bits(std::get<0>(params));

    if (std::isnan(exp_value)) {
        EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
    } else {
        EXPECT_EQ(static_cast<float>(f8), exp_value);
    }
}

TEST_P(F8E5M2PTest, f32_to_f8_bits) {
    const auto& params = GetParam();
    const auto& value = std::get<1>(params);
    auto exp_value = std::get<0>(params);
    const auto f8 = ov::float8_e5m2(value);
    const auto f8_bits = f8.to_bits();

    if (exp_value == 0xFF || exp_value == 0x7F) {  // quiet NaN
        EXPECT_TRUE(std::isnan(f8));
        exp_value--;
    } else if (exp_value == 0x7D || exp_value == 0xFD) {  // signaling NaN
        // signaling NaN representation can be implementation defined
        const auto allowed_nan = (f8_bits == exp_value) || (f8_bits == ++exp_value);
        EXPECT_TRUE(std::isnan(f8));
        EXPECT_TRUE(allowed_nan);
        return;
    }

    EXPECT_EQ(f8_bits, exp_value);
}
}  // namespace test
}  // namespace ov
