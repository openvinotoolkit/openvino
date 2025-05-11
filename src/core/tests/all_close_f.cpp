// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/all_close_f.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <limits>
#include <sstream>

#include "common_test_utils/float_util.hpp"

using namespace std;

class all_close_f_param_test : public testing::TestWithParam<::std::tuple<float, int>> {
protected:
    all_close_f_param_test()
        : upper_bound(FLT_MAX),
          lower_bound(-FLT_MAX),
          past_upper_bound(FLT_MAX),
          past_lower_bound(-FLT_MAX) {
        std::tie(expected, tolerance_bits) = GetParam();
    }
    void SetUp() override {
        constexpr int mantissa_bits = 24;
        uint32_t expected_as_int = ov::test::utils::FloatUnion(expected).i;

        // Turn on targeted bit
        // e.g. for float with 24 bit mantissa, 2 bit accuracy, and hard-coded 8 bit exponent_bits
        // tolerance_bit_shift = 32 -           (1 +  8 + (24 -     1         ) - 2             )
        //                       float_length    sign exp  mantissa implicit 1    tolerance_bits
        uint32_t tolerance_bit_shift = 32 - (1 + 8 + (mantissa_bits - 1) - tolerance_bits);
        uint32_t targeted_bit = (1u << tolerance_bit_shift);

        if (expected > 0.f) {
            uint32_t upper_bound_as_int = expected_as_int + targeted_bit;
            upper_bound = ov::test::utils::FloatUnion(upper_bound_as_int).f;
            past_upper_bound = ov::test::utils::FloatUnion(upper_bound_as_int + 1).f;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::FloatUnion(upper_bound_as_int + 2).f;

            uint32_t lower_bound_as_int = expected_as_int - targeted_bit;
            lower_bound = ov::test::utils::FloatUnion(lower_bound_as_int).f;
            past_lower_bound = ov::test::utils::FloatUnion(lower_bound_as_int - 1).f;
        } else if (expected < 0.f) {
            // Same logic/math as above, but reversed variable name order
            uint32_t lower_bound_as_int = expected_as_int + targeted_bit;
            lower_bound = ov::test::utils::FloatUnion(lower_bound_as_int).f;
            past_lower_bound = ov::test::utils::FloatUnion(lower_bound_as_int + 1).f;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::FloatUnion(lower_bound_as_int + 2).f;

            uint32_t upper_bound_as_int = expected_as_int - targeted_bit;
            upper_bound = ov::test::utils::FloatUnion(upper_bound_as_int).f;
            past_upper_bound = ov::test::utils::FloatUnion(upper_bound_as_int - 1).f;
        } else  // (expected == 0.f) || (expected == -0.f)
        {
            // Special handling of 0 / -0 which get same bounds
            uint32_t upper_bound_as_int = targeted_bit;
            upper_bound = ov::test::utils::FloatUnion(upper_bound_as_int).f;
            uint32_t past_upper_bound_as_int = upper_bound_as_int + 1;
            past_upper_bound = ov::test::utils::FloatUnion(past_upper_bound_as_int).f;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::FloatUnion(upper_bound_as_int + 2).f;

            lower_bound = ov::test::utils::FloatUnion(upper_bound_as_int | 0x80000000).f;
            past_lower_bound = ov::test::utils::FloatUnion(past_upper_bound_as_int | 0x80000000).f;
        }
    }

    float expected{0};
    int tolerance_bits{0};
    float upper_bound;
    float lower_bound;
    float past_upper_bound;
    float past_lower_bound;
    float min_signal_too_low{0};
    float min_signal_enables_passing{0};
};

TEST_P(all_close_f_param_test, test_boundaries) {
    // Format verbose info to only print out in case of test failure
    stringstream ss;
    ss << "Testing target of: " << expected << " (" << ov::test::utils::float_to_bits(expected) << ")\n";
    ss << "Matching to targets with: " << tolerance_bits << " tolerance_bits\n";
    ss << "upper_bound: " << upper_bound << " (" << ov::test::utils::float_to_bits(upper_bound) << ")\n";
    ss << "lower_bound: " << lower_bound << " (" << ov::test::utils::float_to_bits(lower_bound) << ")\n";
    ss << "past_upper_bound: " << past_upper_bound << " (" << ov::test::utils::float_to_bits(past_upper_bound) << ")\n";
    ss << "past_lower_bound: " << past_lower_bound << " (" << ov::test::utils::float_to_bits(past_lower_bound) << ")\n";

    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits)) << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits)) << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits)) << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits, min_signal_too_low)) << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits, min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(
        ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({past_upper_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({past_upper_bound}),
                                              tolerance_bits,
                                              min_signal_too_low))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({past_upper_bound}),
                                             tolerance_bits,
                                             min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits)) << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits, min_signal_too_low)) << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits, min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(
        ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({past_lower_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({past_lower_bound}),
                                              tolerance_bits,
                                              min_signal_too_low))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({past_lower_bound}),
                                             tolerance_bits,
                                             min_signal_enables_passing))
        << ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    test_simple_floats_with_range_of_precisions,
    all_close_f_param_test,
    testing::Combine(
        testing::Values(0.f, -0.f, 1.f, -1.f, 10.f, -10.f, 0.75f, -0.75f, 0.5f, -0.5f, 0.25f, -0.25f, 0.125f, -0.125f),
        testing::Range(0, 5)));

class all_close_f_double_param_test : public testing::TestWithParam<::std::tuple<double, int>> {
protected:
    all_close_f_double_param_test()
        : upper_bound(DBL_MAX),
          lower_bound(-DBL_MAX),
          past_upper_bound(DBL_MAX),
          past_lower_bound(-DBL_MAX) {
        std::tie(expected, tolerance_bits) = GetParam();
    }
    void SetUp() override {
        constexpr int mantissa_bits = 53;
        uint64_t expected_as_int = ov::test::utils::DoubleUnion(expected).i;
        // Turn on targeted bit
        // e.g. for double with 52 bit mantissa, 2 bit accuracy, and hard-coded 11 bit exponent_bits
        // tolerance_bit_shift = 64 -           (1 +  11 + (52 -     1         ) - 2             )
        //                       double_length   sign exp   mantissa implicit 1    tolerance_bits
        uint64_t tolerance_bit_shift = 64 - (1 + 11 + (mantissa_bits - 1) - tolerance_bits);
        uint64_t targeted_bit = (1ull << tolerance_bit_shift);

        if (expected > 0.) {
            uint64_t upper_bound_as_int = expected_as_int + targeted_bit;
            upper_bound = ov::test::utils::DoubleUnion(upper_bound_as_int).d;
            past_upper_bound = ov::test::utils::DoubleUnion(upper_bound_as_int + 1).d;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::DoubleUnion(upper_bound_as_int + 2).d;

            uint64_t lower_bound_as_int = expected_as_int - targeted_bit;
            lower_bound = ov::test::utils::DoubleUnion(lower_bound_as_int).d;
            past_lower_bound = ov::test::utils::DoubleUnion(lower_bound_as_int - 1).d;
        } else if (expected < 0.) {
            // Same logic/math as above, but reversed variable name order
            uint64_t lower_bound_as_int = expected_as_int + targeted_bit;
            lower_bound = ov::test::utils::DoubleUnion(lower_bound_as_int).d;
            past_lower_bound = ov::test::utils::DoubleUnion(lower_bound_as_int + 1).d;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::DoubleUnion(lower_bound_as_int + 2).d;

            uint64_t upper_bound_as_int = expected_as_int - targeted_bit;
            upper_bound = ov::test::utils::DoubleUnion(upper_bound_as_int).d;
            past_upper_bound = ov::test::utils::DoubleUnion(upper_bound_as_int - 1).d;
        } else  // (expected == 0.) || (expected == -0.)
        {
            // Special handling of 0 / -0 which get same bounds
            uint64_t upper_bound_as_int = targeted_bit;
            upper_bound = ov::test::utils::DoubleUnion(upper_bound_as_int).d;
            uint64_t past_upper_bound_as_int = upper_bound_as_int + 1;
            past_upper_bound = ov::test::utils::DoubleUnion(past_upper_bound_as_int).d;
            min_signal_too_low = expected;
            min_signal_enables_passing = ov::test::utils::DoubleUnion(upper_bound_as_int + 2).d;

            lower_bound = ov::test::utils::DoubleUnion(upper_bound_as_int | 0x8000000000000000).d;
            past_lower_bound = ov::test::utils::DoubleUnion(past_upper_bound_as_int | 0x8000000000000000).d;
        }
    }

    double expected{0};
    int tolerance_bits{0};
    double upper_bound;
    double lower_bound;
    double past_upper_bound;
    double past_lower_bound;
    double min_signal_too_low{0};
    double min_signal_enables_passing{0};
};

TEST_P(all_close_f_double_param_test, test_boundaries) {
    // Format verbose info to only print out in case of test failure
    stringstream ss;
    ss << "Testing target of: " << expected << " (" << ov::test::utils::double_to_bits(expected) << ")\n";
    ss << "Matching to targets with: " << tolerance_bits << " tolerance_bits\n";
    ss << "upper_bound: " << upper_bound << " (" << ov::test::utils::double_to_bits(upper_bound) << ")\n";
    ss << "lower_bound: " << lower_bound << " (" << ov::test::utils::double_to_bits(lower_bound) << ")\n";
    ss << "past_upper_bound: " << past_upper_bound << " (" << ov::test::utils::double_to_bits(past_upper_bound)
       << ")\n";
    ss << "past_lower_bound: " << past_lower_bound << " (" << ov::test::utils::double_to_bits(past_lower_bound)
       << ")\n";

    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits)) << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({expected}), vector<double>({upper_bound}), tolerance_bits))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits)) << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({expected}), vector<double>({lower_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits)) << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits, min_signal_too_low)) << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, past_upper_bound, tolerance_bits, min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(
        ov::test::utils::all_close_f(vector<double>({expected}), vector<double>({past_upper_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({expected}),
                                              vector<double>({past_upper_bound}),
                                              tolerance_bits,
                                              min_signal_too_low))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({expected}),
                                             vector<double>({past_upper_bound}),
                                             tolerance_bits,
                                             min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits)) << ss.str();
    EXPECT_FALSE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits, min_signal_too_low)) << ss.str();
    EXPECT_TRUE(ov::test::utils::close_f(expected, past_lower_bound, tolerance_bits, min_signal_enables_passing))
        << ss.str();
    EXPECT_FALSE(
        ov::test::utils::all_close_f(vector<double>({expected}), vector<double>({past_lower_bound}), tolerance_bits))
        << ss.str();
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({expected}),
                                              vector<double>({past_lower_bound}),
                                              tolerance_bits,
                                              min_signal_too_low))
        << ss.str();
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({expected}),
                                             vector<double>({past_lower_bound}),
                                             tolerance_bits,
                                             min_signal_enables_passing))
        << ss.str();
}

INSTANTIATE_TEST_SUITE_P(
    test_simple_doubles_with_range_of_precisions,
    all_close_f_double_param_test,
    testing::Combine(testing::Values(0., -0., 1., -1., 10., -10., 0.75, -0.75, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125),
                     testing::Range(0, 17)));

// Test the exact bounds near +0.f
//
// With tolerance_bits = 18
// (equivalent to testing bfloat precision with 2 bits tolerance)
//
//                           Targeted bit
//                           |
//                           v
//                   2 3 4 5 6    (error allowed in 6th bit or later, w/ implicit leading bit)
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|      8      |    (8 w/ implicit leading bit)
//                           ^
//                           |  2 |<=
//
// [Upper bound]
//                           Add 1 at this bit
//                           |
//                           v
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Convert to 2's compliment
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Mask the sign bit
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_0) {
    constexpr int tolerance_bits = (FLOAT_MANTISSA_BITS - BFLOAT_MANTISSA_BITS + 2);

    // 0.f, the ground-truth value
    float expected = ov::test::utils::bits_to_float("0  00000000  000 0000 0000 0000 0000 0000");
    float computed;
    float min_signal_too_low = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0001");
    float min_signal_enables_passing = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0010");

    // ~3.67342E-40, the exact upper bound
    computed = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // ~3.67343E-40, the next representable number bigger than upper bound
    computed = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_enables_passing));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({computed}),
                                              tolerance_bits,
                                              min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({computed}),
                                             tolerance_bits,
                                             min_signal_enables_passing));

    // ~-3.67342E-40, the exact lower bound
    computed = ov::test::utils::bits_to_float("1  00000000  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = ov::test::utils::bits_to_float("1  00000000  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_enables_passing));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({computed}),
                                              tolerance_bits,
                                              min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({computed}),
                                             tolerance_bits,
                                             min_signal_enables_passing));
}

// Test the exact bounds near -0.f
//
// With tolerance_bits = 18
// (equivalent to testing bfloat precision with 2 bits tolerance)
//
//                           Targeted bit
//                           |
//                           v
//                   2 3 4 5 6    (error allowed in 6th bit or later, w/ implicit leading bit)
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|      8      |    (8 w/ implicit leading bit)
//                           ^
//                           |  2 |<=
//
// [Upper bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Convert to 2's compliment
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// Mask off sign bit
// 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Add 1 at this bit
//                           |
//                           v
// 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_n0) {
    constexpr int tolerance_bits = (FLOAT_MANTISSA_BITS - BFLOAT_MANTISSA_BITS + 2);

    // 0.f, the ground-truth value
    float expected = ov::test::utils::bits_to_float("1  00000000  000 0000 0000 0000 0000 0000");
    float computed;
    float min_signal_too_low = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0001");
    float min_signal_enables_passing = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0010");

    // ~3.67342E-40, the exact upper bound
    computed = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // ~3.67343E-40, the next representable number bigger than upper bound
    computed = ov::test::utils::bits_to_float("0  00000000  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_enables_passing));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({computed}),
                                              tolerance_bits,
                                              min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({computed}),
                                             tolerance_bits,
                                             min_signal_enables_passing));

    // ~-3.67342E-40, the exact lower bound
    computed = ov::test::utils::bits_to_float("1  00000000  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // ~-3.67343E-40, the next representable number smaller than lower bound
    computed = ov::test::utils::bits_to_float("1  00000000  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits, min_signal_enables_passing));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({computed}),
                                              tolerance_bits,
                                              min_signal_too_low));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}),
                                             vector<float>({computed}),
                                             tolerance_bits,
                                             min_signal_enables_passing));
}

// Test the exact bounds near 1.f
//
// With tolerance_bits = 18
// (equivalent to testing bfloat precision with 2 bits tolerance)
//
//                           Targeted bit
//                           |
//                           v
//                   2 3 4 5 6    (error allowed in 6th bit or later, w/ implicit leading bit)
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|      8      |    (8 w/ implicit leading bit)
//                           ^
//                           |  2 |<=
//
// [Upper bound]
//                           Add 1 at this bit to get upper bound
//                           |
//                           v
// 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Minus 1 at this bit to get lower bound
//                           |
//                           v
// 0 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 0 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_1) {
    constexpr int tolerance_bits = (FLOAT_MANTISSA_BITS - BFLOAT_MANTISSA_BITS + 2);

    // 1.f, the ground-truth value
    float expected = ov::test::utils::bits_to_float("0  01111111  000 0000 0000 0000 0000 0000");
    float computed;

    // 1.03125f, the exact upper bound
    computed = ov::test::utils::bits_to_float("0  01111111  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // 1.031250119f, the next representable number bigger than upper bound
    computed = ov::test::utils::bits_to_float("0  01111111  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // 0.984375f, the exact lower bound
    computed = ov::test::utils::bits_to_float("0  01111110  111 1100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // 0.9843749404f, the next representable number smaller than lower bound
    computed = ov::test::utils::bits_to_float("0  01111110  111 1011 1111 1111 1111 1111");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
}

// Test the exact bounds near -1.f
//
// With tolerance_bits = 18
// (equivalent to testing bfloat precision with 2 bits tolerance)
//
//                           Targeted bit
//                           |
//                           v
//                   2 3 4 5 6    (error allowed in 6th bit or later, w/ implicit leading bit)
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|      8      |    (8 w/ implicit leading bit)
//                           ^
//                           |  2 |<=
//
// [Upper bound]
//                           Minus 1 at this bit
//                           |
//                           v
// 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// -                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 1 1 1 1 1 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
//
// [Lower bound]
//                           Add 1 at this bit
//                           |
//                           v
// 1 0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// +                         1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
// ---------------------------------------------------------------
// 1 0 1 1 1 1 1 1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
TEST(all_close_f, mantissa_8_near_n1) {
    constexpr int tolerance_bits = (FLOAT_MANTISSA_BITS - BFLOAT_MANTISSA_BITS + 2);

    // -1.f, the ground-truth value
    float expected = ov::test::utils::bits_to_float("1  01111111  000 0000 0000 0000 0000 0000");
    float computed;

    // -0.984375f, the exact upper bound
    computed = ov::test::utils::bits_to_float("1  01111110  111 1100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // -0.984374940395355224609375f, the next representable number bigger than upper bound
    computed = ov::test::utils::bits_to_float("1  01111110  111 1011 1111 1111 1111 1111");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // -1.03125f, the exact lower bound
    computed = ov::test::utils::bits_to_float("1  01111111  000 0100 0000 0000 0000 0000");
    EXPECT_TRUE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));

    // -1.03125011920928955078125f, the next representable number smaller than lower bound
    computed = ov::test::utils::bits_to_float("1  01111111  000 0100 0000 0000 0000 0001");
    EXPECT_FALSE(ov::test::utils::close_f(expected, computed, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({computed}), tolerance_bits));
}

// For intuitive understanding of tightness of bounds in decimal
// Test bounds near 0, 1, 10, 100, 1000 with tolerance_bits = 18
//
//                           Targeted bit
//                           |
//                           v
//                   2 3 4 5 6    (error allowed in 6th bit or later, w/ implicit leading bit)
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|      8      |    (8 w/ implicit leading bit)
//                           ^
//                           |  2 |<=
TEST(all_close_f, mantissa_8_near_0_1_10_100_1000) {
    constexpr int tolerance_bits = (FLOAT_MANTISSA_BITS - BFLOAT_MANTISSA_BITS + 2);

    float expected;
    float upper_bound;
    float bigger_than_upper_bound;
    float lower_bound;
    float smaller_than_lower_bound;

    // Bounds around 0: 0 +- 3.67e-40
    expected = 0.f;                           // 0  00000000  000 0000 0000 0000 0000 0000
    upper_bound = 3.67342e-40f;               // 0  00000000  000 0100 0000 0000 0000 0000, approximated
    bigger_than_upper_bound = 3.67343e-40f;   // 0  00000000  000 0100 0000 0000 0000 0001, approximated
    lower_bound = -3.67342e-40f;              // 1  00000000  000 0100 0000 0000 0000 0000, approximated
    smaller_than_lower_bound = 3.67343e-40f;  // 1  00000000  000 0100 0000 0000 0000 0001, approximated
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 1: 1 +- 0.03
    expected = 1.f;                            // 0  01111111  000 0000 0000 0000 0000 0000
    upper_bound = 1.03125f;                    // 0  01111111  000 0100 0000 0000 0000 0000
    bigger_than_upper_bound = 1.031250119f;    // 0  01111111  000 0100 0000 0000 0000 0001
    lower_bound = 0.984375f;                   // 0  01111110  111 1100 0000 0000 0000 0000
    smaller_than_lower_bound = 0.9843749404f;  // 0  01111110  111 1011 1111 1111 1111 1111
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 10: 10 +- 0.25
    expected = 10.f;                                     // 0  10000010  010 0000 0000 0000 0000 0000
    upper_bound = 10.25f;                                // 0  10000010  010 0100 0000 0000 0000 0000
    bigger_than_upper_bound = 10.25000095367431640625f;  // 0  10000010  010 0100 0000 0000 0000 0001
    lower_bound = 9.75f;                                 // 0  10000010  001 1100 0000 0000 0000 0000
    smaller_than_lower_bound = 9.74999904632568359375f;  // 0  10000010  001 1011 1111 1111 1111 1111
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 100: 100 +- 2
    expected = 100.f;                                  // 0  10000101  100 1000 0000 0000 0000 0000
    upper_bound = 102.f;                               // 0  10000101  100 1100 0000 0000 0000 0000
    bigger_than_upper_bound = 102.00000762939453125f;  // 0  10000101  100 1100 0000 0000 0000 0001
    lower_bound = 98.0f;                               // 0  10000101  100 0100 0000 0000 0000 0000
    smaller_than_lower_bound = 97.99999237060546875f;  // 0  10000101  100 0011 1111 1111 1111 1111
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 1000: 1000 +- 16
    expected = 1000.f;                               // 0  10001000  111 1010 0000 0000 0000 0000
    upper_bound = 1016.f;                            // 0  10001000  111 1110 0000 0000 0000 0000
    bigger_than_upper_bound = 1016.00006103515625f;  // 0  10001000  111 1110 0000 0000 0000 0001
    lower_bound = 984.0f;                            // 0  10001000  111 0110 0000 0000 0000 0000
    smaller_than_lower_bound = 983.99993896484375f;  // 0  10001000  111 0101 1111 1111 1111 1111
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));
}

// For intuitive understanding of tightness of bounds in decimal
// Test bounds near 0, 1, 10, 100, 1000 with tolerance_bits = 2
//
//                                                           Targeted bit
//                                                           |
//            (22 bits must match, w/ implicit leading bit)  v
//                   2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2
// s e e e e e e e e m m m m m m m m m m m m m m m m m m m m m m m
//                =>|        24 (w/ implicit leading bit)         |
//                                                           ^
//                                                           | 2  |<=
TEST(all_close_f, mantissa_24_near_0_1_10_100_1000) {
    constexpr int tolerance_bits = 2;

    float expected;
    float upper_bound;
    float bigger_than_upper_bound;
    float lower_bound;
    float smaller_than_lower_bound;

    // Bounds around 0: 0 +- 5.6e-45
    expected = 0.f;
    upper_bound = ov::test::utils::bits_to_float("0  00000000  000 0000 0000 0000 0000 0100");
    bigger_than_upper_bound = ov::test::utils::bits_to_float("0  00000000  000 0000 0000 0000 0000 0101");
    lower_bound = ov::test::utils::bits_to_float("1  00000000  000 0000 0000 0000 0000 0100");
    smaller_than_lower_bound = ov::test::utils::bits_to_float("1  00000000  000 0000 0000 0000 0000 0101");
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 1: 1 +- 4.77e-7
    expected = 1.f;
    upper_bound = ov::test::utils::bits_to_float("0  01111111  000 0000 0000 0000 0000 0100");
    bigger_than_upper_bound = ov::test::utils::bits_to_float("0  01111111  000 0000 0000 0000 0000 0101");
    lower_bound = ov::test::utils::bits_to_float("0  01111110  111 1111 1111 1111 1111 1100");
    smaller_than_lower_bound = ov::test::utils::bits_to_float("0  01111110  111 1111 1111 1111 1111 1011");
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 10: 10 +- 3.81e-6
    expected = 10.f;
    upper_bound = ov::test::utils::bits_to_float("0  10000010  010 0000 0000 0000 0000 0100");
    bigger_than_upper_bound = ov::test::utils::bits_to_float("0  10000010  010 0000 0000 0000 0000 0101");
    lower_bound = ov::test::utils::bits_to_float("0  10000010  001 1111 1111 1111 1111 1100");
    smaller_than_lower_bound = ov::test::utils::bits_to_float("0  10000010  001 1111 1111 1111 1111 1011");
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 100: 100 +- 3.05e-5
    expected = 100.f;
    upper_bound = ov::test::utils::bits_to_float("0  10000101  100 1000 0000 0000 0000 0100");
    bigger_than_upper_bound = ov::test::utils::bits_to_float("0  10000101  100 1000 0000 0000 0000 0101");
    lower_bound = ov::test::utils::bits_to_float("0  10000101  100 0111 1111 1111 1111 1100");
    smaller_than_lower_bound = ov::test::utils::bits_to_float("0  10000101  100 0111 1111 1111 1111 1011");
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));

    // Bounds around 1000: 1000 +- 2.44e-4
    expected = 1000.f;
    upper_bound = ov::test::utils::bits_to_float("0  10001000  111 1010 0000 0000 0000 0100");
    bigger_than_upper_bound = ov::test::utils::bits_to_float("0  10001000  111 1010 0000 0000 0000 0101");
    lower_bound = ov::test::utils::bits_to_float("0  10001000  111 1001 1111 1111 1111 1100");
    smaller_than_lower_bound = ov::test::utils::bits_to_float("0  10001000  111 1001 1111 1111 1111 1011");
    EXPECT_TRUE(ov::test::utils::close_f(expected, upper_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({upper_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, bigger_than_upper_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({bigger_than_upper_bound}),
                                              tolerance_bits));
    EXPECT_TRUE(ov::test::utils::close_f(expected, lower_bound, tolerance_bits));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({expected}), vector<float>({lower_bound}), tolerance_bits));
    EXPECT_FALSE(ov::test::utils::close_f(expected, smaller_than_lower_bound, tolerance_bits));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({expected}),
                                              vector<float>({smaller_than_lower_bound}),
                                              tolerance_bits));
}

TEST(all_close_f, inf_nan) {
    float zero = 0.f;
    float infinity = numeric_limits<float>::infinity();
    float neg_infinity = -numeric_limits<float>::infinity();
    float quiet_nan = numeric_limits<float>::quiet_NaN();
    float signaling_nan = numeric_limits<float>::signaling_NaN();

    EXPECT_FALSE(ov::test::utils::close_f(zero, infinity));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({zero}), vector<float>({infinity})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, neg_infinity));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({zero}), vector<float>({neg_infinity})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, quiet_nan));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({zero}), vector<float>({quiet_nan})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, signaling_nan));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<float>({zero}), vector<float>({signaling_nan})));

    EXPECT_TRUE(ov::test::utils::close_f(infinity, infinity));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({infinity}), vector<float>({infinity})));
    EXPECT_TRUE(ov::test::utils::close_f(neg_infinity, neg_infinity));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({neg_infinity}), vector<float>({neg_infinity})));
    EXPECT_TRUE(ov::test::utils::close_f(quiet_nan, quiet_nan));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({quiet_nan}), vector<float>({quiet_nan})));
    EXPECT_TRUE(ov::test::utils::close_f(signaling_nan, signaling_nan));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<float>({signaling_nan}), vector<float>({signaling_nan})));
}

TEST(all_close_f, double_inf_nan) {
    double zero = 0;
    double infinity = numeric_limits<double>::infinity();
    double neg_infinity = -numeric_limits<double>::infinity();
    double quiet_nan = numeric_limits<double>::quiet_NaN();
    double signaling_nan = numeric_limits<double>::signaling_NaN();

    EXPECT_FALSE(ov::test::utils::close_f(zero, infinity));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({zero}), vector<double>({infinity})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, neg_infinity));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({zero}), vector<double>({neg_infinity})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, quiet_nan));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({zero}), vector<double>({quiet_nan})));
    EXPECT_FALSE(ov::test::utils::close_f(zero, signaling_nan));
    EXPECT_FALSE(ov::test::utils::all_close_f(vector<double>({zero}), vector<double>({signaling_nan})));

    EXPECT_TRUE(ov::test::utils::close_f(infinity, infinity));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({infinity}), vector<double>({infinity})));
    EXPECT_TRUE(ov::test::utils::close_f(neg_infinity, neg_infinity));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({neg_infinity}), vector<double>({neg_infinity})));
    EXPECT_TRUE(ov::test::utils::close_f(quiet_nan, quiet_nan));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({quiet_nan}), vector<double>({quiet_nan})));
    EXPECT_TRUE(ov::test::utils::close_f(signaling_nan, signaling_nan));
    EXPECT_TRUE(ov::test::utils::all_close_f(vector<double>({signaling_nan}), vector<double>({signaling_nan})));
}
