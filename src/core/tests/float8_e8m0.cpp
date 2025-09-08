// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/float8_e8m0.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <tuple>

#include "common_test_utils/float_util.hpp"

namespace ov {
namespace test {

TEST(F8E8M0Test, stream_operator) {
    std::stringstream s;
    s << ov::float8_e8m0(2.0f);

    EXPECT_EQ(s.str(), "2");
}

TEST(F8E8M0Test, to_string) {
    const auto f8 = ov::float8_e8m0::from_bits(0b10010111);

    EXPECT_EQ(std::to_string(f8), "16777216.000000");
}

TEST(F8E8M0Test, to_float) {
    const auto f8 = ov::float8_e8m0::from_bits(0b00000001);

    EXPECT_EQ(static_cast<float>(f8), 1.1754944e-38f);
}

TEST(F8E8M0Test, f32_inf) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<float>::infinity());

    EXPECT_EQ(f8.to_bits(), 0b11111110);
}

TEST(F8E8M0Test, f32_minus_inf) {
    const auto f8 = ov::float8_e8m0(-std::numeric_limits<float>::infinity());

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(F8E8M0Test, f32_nan_quiet) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<float>::quiet_NaN());

    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f32_nan_signaling) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<float>::signaling_NaN());

    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f32_zero) {
    const auto f8 = ov::float8_e8m0(0.0f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(F8E8M0Test, f32_negative_zero) {
    const auto f8 = ov::float8_e8m0(-0.0f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(F8E8M0Test, f32_negative_one) {
    const auto f8 = ov::float8_e8m0(-1.0f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
}

TEST(F8E8M0Test, F8E8M0_nan) {
    const auto f8 = static_cast<float>(std::numeric_limits<ov::float8_e8m0>::quiet_NaN());

    EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
}

TEST(F8E8M0Test, f8e8m0_num_limits_is_specialized) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::is_specialized;
    EXPECT_TRUE(val);
}

TEST(F8E8M0Test, f8e8m0_num_limits_is_signed) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::is_signed;
    EXPECT_FALSE(val);
}

TEST(F8E8M0Test, f8e8m0_num_limits_is_integer) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::is_integer;
    EXPECT_FALSE(val);
}

TEST(F8E8M0Test, f8e8m0_num_limits_is_exact) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::is_exact;
    EXPECT_FALSE(val);
}

TEST(F8E8M0Test, f8e8m0_num_limits_radix) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::radix;
    EXPECT_EQ(val, 2);
}

TEST(F8E8M0Test, f8e8m0_num_limits_digits) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::digits;
    EXPECT_EQ(val, 1);
}

TEST(F8E8M0Test, f8e8m0_num_limits_digits10) {
    const auto f8_dig = std::numeric_limits<ov::float8_e8m0>::digits;
    const auto f8_dig10 = std::numeric_limits<ov::float8_e8m0>::digits10;

    EXPECT_EQ(f8_dig10, static_cast<int>((f8_dig - 1) * std::log10(2)));
    EXPECT_EQ(f8_dig10, 0);
}

TEST(F8E8M0Test, f8e8m0_num_limits_epsilon) {
    const auto f8_1 = ov::float8_e8m0(1.f);
    const auto f8_1_bits = f8_1.to_bits();
    const auto f8_1_next_bits = f8_1_bits + 1u;

    const auto f8_eps = ov::float8_e8m0::from_bits(f8_1_next_bits - f8_1_bits);

    EXPECT_EQ(f8_eps, std::numeric_limits<ov::float8_e8m0>::epsilon());
    EXPECT_EQ(f8_eps.to_bits(), std::numeric_limits<ov::float8_e8m0>::epsilon().to_bits());
}

TEST(F8E8M0Test, f8e8m0_num_limits_round_error) {
    const auto f8 = ov::float8_e8m0(0.5f);

    EXPECT_EQ(f8, std::numeric_limits<ov::float8_e8m0>::round_error());
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e8m0>::round_error().to_bits());
}

TEST(F8E8M0Test, f8e8m0_quiet_nan) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<ov::float8_e8m0>::quiet_NaN());
    const auto has_quiet_nan = std::numeric_limits<ov::float8_e8m0>::has_quiet_NaN;

    EXPECT_TRUE(has_quiet_nan);
    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
    EXPECT_EQ(std::numeric_limits<ov::float8_e8m0>::quiet_NaN().to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f8e8m0_sig_nan) {
    const auto val = std::numeric_limits<ov::float8_e8m0>::has_signaling_NaN;
    EXPECT_FALSE(val);
}

TEST(F8E8M0Test, f16_quiet_nan) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<ov::float16>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
    EXPECT_EQ(std::numeric_limits<ov::float8_e8m0>::quiet_NaN().to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f16_sig_nan) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<ov::float16>::signaling_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f32_quiet_nan) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<float>::quiet_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f32_sig_nan) {
    const auto f8 = ov::float8_e8m0(std::numeric_limits<float>::signaling_NaN());

    EXPECT_TRUE(std::isnan(f8));
    EXPECT_EQ(f8.to_bits(), 0b11111111);
}

TEST(F8E8M0Test, f8e8m0_min_normalized) {
    const auto f8 = ov::float8_e8m0(5.877471754111437e-39f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e8m0>::min().to_bits());
}

TEST(F8E8M0Test, f8e8m0_max_normalized) {
    const auto f8 = ov::float8_e8m0(1.7014118346046923e+38f);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e8m0>::max().to_bits());
}

TEST(F8E8M0Test, f8e8m0_lowest_normalized) {
    const auto f8 = ov::float8_e8m0(5.877471754111437e-39f);

    EXPECT_EQ(f8.to_bits(), 0b00000000);
    EXPECT_EQ(f8.to_bits(), std::numeric_limits<ov::float8_e8m0>::lowest().to_bits());
}

TEST(F8E8M0Test, f32_ge_f8_round_to_max) {
    const auto f8 = ov::float8_e8m0(3.402823466385288598117042E+38);

    EXPECT_EQ(f8.to_bits(), 0b11111110);
}

TEST(F8E8M0Test, f8e8m0_num_limits_exp) {
    const auto min_exp = std::numeric_limits<ov::float8_e8m0>::min_exponent;
    const auto min_exp10 = std::numeric_limits<ov::float8_e8m0>::min_exponent10;
    const auto max_exp = std::numeric_limits<ov::float8_e8m0>::max_exponent;
    const auto max_exp10 = std::numeric_limits<ov::float8_e8m0>::max_exponent10;

    EXPECT_EQ(min_exp, -126);
    EXPECT_EQ(min_exp10, -38);
    EXPECT_EQ(max_exp, 128);
    EXPECT_EQ(max_exp10, 38);
}

template <class TContainer>
std::vector<std::tuple<uint8_t, typename TContainer::value_type>> enumerate(const TContainer& values) {
    std::vector<std::tuple<uint8_t, typename TContainer::value_type>> enum_values;
    uint8_t i = 0;
    for (const auto& v : values) {
        enum_values.emplace_back(i++, v);
    }
    return enum_values;
}

constexpr auto f32_nan = std::numeric_limits<float>::quiet_NaN();
constexpr float two_power_127 = 1.7014118346046923e+38f;
constexpr float two_power_negative_127 = 5.877471754111437e-39f;

// clang-format off
const auto exp_floats = std::vector<float>{
    two_power_negative_127,     1.1754943508222875e-38f,    2.350988701644575e-38f,     4.70197740328915e-38f,
    9.4039548065783e-38f,       1.88079096131566e-37f,      3.76158192263132e-37f,      7.52316384526264e-37f,
    1.504632769052528e-36f,     3.009265538105056e-36f,     6.018531076210112e-36f,     1.2037062152420224e-35f,
    2.407412430484045e-35f,     4.81482486096809e-35f,      9.62964972193618e-35f,      1.925929944387236e-34f,
    3.851859888774472e-34f,     7.703719777548943e-34f,     1.5407439555097887e-33f,    3.0814879110195774e-33f,
    6.162975822039155e-33f,     1.232595164407831e-32f,     2.465190328815662e-32f,     4.930380657631324e-32f,
    9.860761315262648e-32f,     1.9721522630525295e-31f,    3.944304526105059e-31f,     7.888609052210118e-31f,
    1.5777218104420236e-30f,    3.1554436208840472e-30f,    6.310887241768095e-30f,     1.262177448353619e-29f,
    2.524354896707238e-29f,     5.048709793414476e-29f,     1.0097419586828951e-28f,    2.0194839173657902e-28f,
    4.0389678347315804e-28f,    8.077935669463161e-28f,     1.6155871338926322e-27f,    3.2311742677852644e-27f,
    6.462348535570529e-27f,     1.2924697071141057e-26f,    2.5849394142282115e-26f,    5.169878828456423e-26f,
    1.0339757656912846e-25f,    2.0679515313825692e-25f,    4.1359030627651384e-25f,    8.271806125530277e-25f,
    1.6543612251060553e-24f,    3.308722450212111e-24f,     6.617444900424222e-24f,     1.3234889800848443e-23f,
    2.6469779601696886e-23f,    5.293955920339377e-23f,     1.0587911840678754e-22f,    2.117582368135751e-22f,
    4.235164736271502e-22f,     8.470329472543003e-22f,     1.6940658945086007e-21f,    3.3881317890172014e-21f,
    6.776263578034403e-21f,     1.3552527156068805e-20f,    2.710505431213761e-20f,     5.421010862427522e-20f,
    1.0842021724855044e-19f,    2.168404344971009e-19f,     4.336808689942018e-19f,     8.673617379884035e-19f,
    1.734723475976807e-18f,     3.469446951953614e-18f,     6.938893903907228e-18f,     1.3877787807814457e-17f,
    2.7755575615628914e-17f,    5.551115123125783e-17f,     1.1102230246251565e-16f,    2.220446049250313e-16f,
    4.440892098500626e-16f,     8.881784197001252e-16f,     1.7763568394002505e-15f,    3.552713678800501e-15f,
    7.105427357601002e-15f,     1.4210854715202004e-14f,    2.842170943040401e-14f,     5.684341886080802e-14f,
    1.1368683772161603e-13f,    2.2737367544323206e-13f,    4.547473508864641e-13f,     9.094947017729282e-13f,
    1.8189894035458565e-12f,    3.637978807091713e-12f,     7.275957614183426e-12f,     1.4551915228366852e-11f,
    2.9103830456733704e-11f,    5.820766091346741e-11f,     1.1641532182693481e-10f,    2.3283064365386963e-10f,
    4.656612873077393e-10f,     9.313225746154785e-10f,     1.862645149230957e-09f,     3.725290298461914e-09f,
    7.450580596923828e-09f,     1.4901161193847656e-08f,    2.9802322387695312e-08f,    5.960464477539063e-08f,
    1.1920928955078125e-07f,    2.384185791015625e-07f,     4.76837158203125e-07f,      9.5367431640625e-07f,
    1.9073486328125e-06f,       3.814697265625e-06f,        7.62939453125e-06f,         1.52587890625e-05f,
    3.0517578125e-05f,          6.103515625e-05f,           0.0001220703125f,           0.000244140625f,
    0.00048828125f,             0.0009765625f,              0.001953125f,               0.00390625f,
    0.0078125f,                 0.015625f,                  0.03125f,                   0.0625f,
    0.125f,                     0.25f,                      0.5f,                       1.0f,
    2.0f,                       4.0f,                       8.0f,                       16.0f,
    32.0f,                      64.0f,                      128.0f,                     256.0f,
    512.0f,                     1024.0f,                    2048.0f,                    4096.0f,
    8192.0f,                    16384.0f,                   32768.0f,                   65536.0f,
    131072.0f,                  262144.0f,                  524288.0f,                  1048576.0f,
    2097152.0f,                 4194304.0f,                 8388608.0f,                 16777216.0f,
    33554432.0f,                67108864.0f,                134217728.0f,               268435456.0f,
    536870912.0f,               1073741824.0f,              2147483648.0f,              4294967296.0f,
    8589934592.0f,              17179869184.0f,             34359738368.0f,             68719476736.0f,
    137438953472.0f,            274877906944.0f,            549755813888.0f,            1099511627776.0f,
    2199023255552.0f,           4398046511104.0f,           8796093022208.0f,           17592186044416.0f,
    35184372088832.0f,          70368744177664.0f,          140737488355328.0f,         281474976710656.0f,
    562949953421312.0f,         1125899906842624.0f,        2251799813685248.0f,        4503599627370496.0f,
    9007199254740992.0f,        1.8014398509481984e+16f,    3.602879701896397e+16f,     7.205759403792794e+16f,
    1.4411518807585587e+17f,    2.8823037615171174e+17f,    5.764607523034235e+17f,     1.152921504606847e+18f,
    2.305843009213694e+18f,     4.611686018427388e+18f,     9.223372036854776e+18f,     1.8446744073709552e+19f,
    3.6893488147419103e+19f,    7.378697629483821e+19f,     1.4757395258967641e+20f,    2.9514790517935283e+20f,
    5.902958103587057e+20f,     1.1805916207174113e+21f,    2.3611832414348226e+21f,    4.722366482869645e+21f,
    9.44473296573929e+21f,      1.888946593147858e+22f,     3.777893186295716e+22f,     7.555786372591432e+22f,
    1.5111572745182865e+23f,    3.022314549036573e+23f,     6.044629098073146e+23f,     1.2089258196146292e+24f,
    2.4178516392292583e+24f,    4.835703278458517e+24f,     9.671406556917033e+24f,     1.9342813113834067e+25f,
    3.8685626227668134e+25f,    7.737125245533627e+25f,     1.5474250491067253e+26f,    3.094850098213451e+26f,
    6.189700196426902e+26f,     1.2379400392853803e+27f,    2.4758800785707605e+27f,    4.951760157141521e+27f,
    9.903520314283042e+27f,     1.9807040628566084e+28f,    3.961408125713217e+28f,     7.922816251426434e+28f,
    1.5845632502852868e+29f,    3.1691265005705735e+29f,    6.338253001141147e+29f,     1.2676506002282294e+30f,
    2.535301200456459e+30f,     5.070602400912918e+30f,     1.0141204801825835e+31f,    2.028240960365167e+31f,
    4.056481920730334e+31f,     8.112963841460668e+31f,     1.6225927682921336e+32f,    3.2451855365842673e+32f,
    6.490371073168535e+32f,     1.298074214633707e+33f,     2.596148429267414e+33f,     5.192296858534828e+33f,
    1.0384593717069655e+34f,    2.076918743413931e+34f,     4.153837486827862e+34f,     8.307674973655724e+34f,
    1.661534994731145e+35f,     3.32306998946229e+35f,      6.64613997892458e+35f,      1.329227995784916e+36f,
    2.658455991569832e+36f,     5.316911983139664e+36f,     1.0633823966279327e+37f,    2.1267647932558654e+37f,
    4.253529586511731e+37f,     8.507059173023462e+37f,     two_power_127,              f32_nan,
};
// clang-format on

using f8m8e0_params = std::tuple<uint8_t, float>;
class F8E8M0PTest : public testing::TestWithParam<f8m8e0_params> {};

INSTANTIATE_TEST_SUITE_P(convert,
                         F8E8M0PTest,
                         testing::ValuesIn(enumerate(exp_floats)),
                         testing::PrintToStringParamName());

TEST_P(F8E8M0PTest, f8_bits_to_f32) {
    const auto& params = GetParam();
    const auto& exp_value = std::get<1>(params);
    const auto f8 = ov::float8_e8m0::from_bits(std::get<0>(params));

    if (std::isnan(exp_value)) {
        EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
    } else {
        EXPECT_EQ(static_cast<float>(f8), exp_value);
    }
}

TEST_P(F8E8M0PTest, f32_to_f8_bits) {
    const auto& params = GetParam();
    const auto& value = std::get<1>(params);
    const auto& exp_value = std::get<0>(params);
    const auto f8 = ov::float8_e8m0(value);
    const auto f8_bits = f8.to_bits();

    if (exp_value == 0xFF) {
        EXPECT_TRUE(std::isnan(f8));
    } else {
        EXPECT_EQ(f8_bits, exp_value);
    }
}

// clang-format off
using f8m8e0_params_cast = std::tuple<float, float>;
const auto test_values = testing::Values<f8m8e0_params_cast>(
    std::make_tuple(-std::numeric_limits<float>::infinity(),                                two_power_negative_127),
    std::make_tuple(-15466.5486f,                                                           two_power_negative_127),
    std::make_tuple(-1.f,                                                                   two_power_negative_127),
    std::make_tuple(0.f,                                                                    two_power_negative_127),
    std::make_tuple(std::nextafter(two_power_127, std::numeric_limits<float>::infinity()),  two_power_127),
    std::make_tuple(std::numeric_limits<float>::max(),                                      two_power_127),
    std::make_tuple(std::numeric_limits<float>::infinity(),                                 two_power_127),
    std::make_tuple(std::numeric_limits<float>::signaling_NaN(),                            f32_nan),
    std::make_tuple(1.f, 1.f),      std::make_tuple(1.49f, 1.f),    std::make_tuple(1.5f, 2.f),     std::make_tuple(1.51f, 2.f),
    std::make_tuple(2.f, 2.f),      std::make_tuple(2.5f, 2.f),     std::make_tuple(3.f, 2.f),      std::make_tuple(3.5f, 4.f),
    std::make_tuple(4.f, 4.f),      std::make_tuple(5.f, 4.f),      std::make_tuple(5.5f, 4.f),     std::make_tuple(5.99f, 4.f),
    std::make_tuple(6.f, 8.f),      std::make_tuple(7.f, 8.f),      std::make_tuple(8.f, 8.f),      std::make_tuple(9.f, 8.f),
    std::make_tuple(10.f, 8.f),     std::make_tuple(11.f, 8.f),     std::make_tuple(12.f, 8.f),     std::make_tuple(13.f, 16.f),
    std::make_tuple(14.f, 16.f),    std::make_tuple(15.f, 16.f),    std::make_tuple(16.f, 16.f),    std::make_tuple(24.f, 32.f),
    std::make_tuple(32.f, 32.f),    std::make_tuple(32.1f, 32.f),   std::make_tuple(63.9f, 64.f),   std::make_tuple(64.f, 64.f));
// clang-format on

class F8E8M0PTestCast : public testing::TestWithParam<f8m8e0_params_cast> {};

INSTANTIATE_TEST_SUITE_P(cast, F8E8M0PTestCast, test_values, testing::PrintToStringParamName());

TEST_P(F8E8M0PTestCast, f32_to_f8_to_f32) {
    const auto& params = GetParam();

    const auto f8 = ov::float8_e8m0(std::get<0>(params));
    const auto& exp_value = std::get<1>(params);

    if (std::isnan(exp_value)) {
        EXPECT_TRUE(std::isnan(static_cast<float>(f8)));
    } else {
        EXPECT_EQ(static_cast<float>(f8), exp_value);
    }
}

}  // namespace test
}  // namespace ov
