// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace tests {

template <typename RealType>
struct uniform_quantized_real_distribution_test : ::testing::Test
{
protected:
    /// @brief Number of iterations of random number generations (and checks).
    static constexpr std::size_t rnd_iter_num = 32768;

    /// @brief Type of distribution used.
    using uqr_dist = typename std::conditional<!std::is_same<RealType, void>::value,
                                               distributions::uniform_quantized_real_distribution<RealType>,
                                               distributions::uniform_quantized_real_distribution<>>::type;
    /// @brief Type of parameter set of distribution used.
    using uqr_dist_param = typename uqr_dist::param_type;
    /// @brief Expected result_type of uniform_quantized_real_distribution.
    using expected_uqr_dist_rt = typename std::conditional<!std::is_same<RealType, void>::value, RealType, float>::type;
};

using uniform_quantized_real_distribution_test_types = ::testing::Types<void, float, double, long double>;
TYPED_TEST_SUITE(uniform_quantized_real_distribution_test, uniform_quantized_real_distribution_test_types);

TYPED_TEST(uniform_quantized_real_distribution_test, param_construct_default)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(1);
    const unsigned expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist_param dist_param_instance1;
    using actual_uqr_dist_rt = typename decltype(dist_param_instance1)::distribution_type::result_type;

    ASSERT_TRUE((std::is_same<actual_uqr_dist_rt, expected_uqr_dist_rt>::value));
    ASSERT_EQ(dist_param_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance1.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_construct_a_b_srb)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    // Any
    auto expected_a       = static_cast<expected_uqr_dist_rt>(-130);
    auto expected_b       = static_cast<expected_uqr_dist_rt>(244);
    unsigned expected_srb = 1U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_param_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance1.significand_rand_bits(), expected_srb);

    // Zero
    expected_a   = static_cast<expected_uqr_dist_rt>(57);
    expected_b   = static_cast<expected_uqr_dist_rt>(73);
    expected_srb = 0U;

    uqr_dist_param dist_param_instance2(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);

    // Almost Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(-65);
    expected_b   = static_cast<expected_uqr_dist_rt>(-45);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 2U;

    uqr_dist_param dist_param_instance3(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_param_instance3.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance3.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance3.significand_rand_bits(), expected_srb);

    // Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(0);
    expected_b   = static_cast<expected_uqr_dist_rt>(0);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist_param dist_param_instance4(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_param_instance4.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance4.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance4.significand_rand_bits(), expected_srb);

    // Over Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(-4);
    expected_b   = static_cast<expected_uqr_dist_rt>(-1);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    unsigned test_srb = expected_srb + 2U;

    uqr_dist_param dist_param_instance5(expected_a, expected_b, test_srb);

    ASSERT_EQ(dist_param_instance5.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance5.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance5.significand_rand_bits(), expected_srb);

    // Throw std::invalid_argument (a > b)
    expected_a   = static_cast<expected_uqr_dist_rt>(40);
    expected_b   = static_cast<expected_uqr_dist_rt>(39);
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist_param dist_param_instance6(expected_a, expected_b, test_srb);
    }, std::invalid_argument);

    // Throw std::invalid_argument (a is infinite)
    expected_a   = -std::numeric_limits<expected_uqr_dist_rt>::infinity();
    expected_b   = static_cast<expected_uqr_dist_rt>(39);
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist_param dist_param_instance7(expected_a, expected_b, test_srb);
    }, std::invalid_argument);

    // Throw std::invalid_argument (b is infinite)
    expected_a   = static_cast<expected_uqr_dist_rt>(40);
    expected_b   = std::numeric_limits<expected_uqr_dist_rt>::infinity();
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist_param dist_param_instance8(expected_a, expected_b, test_srb);
    }, std::invalid_argument);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_construct_srb)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b = static_cast<expected_uqr_dist_rt>(1);

    // Any
    unsigned expected_srb = 4U;

    uqr_dist_param dist_param_instance1(expected_srb);

    ASSERT_EQ(dist_param_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance1.significand_rand_bits(), expected_srb);

    // Zero
    expected_srb = 0U;

    uqr_dist_param dist_param_instance2(expected_srb);

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);

    // Almost Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 4U;

    uqr_dist_param dist_param_instance3(expected_srb);

    ASSERT_EQ(dist_param_instance3.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance3.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance3.significand_rand_bits(), expected_srb);

    // Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist_param dist_param_instance4(expected_srb);

    ASSERT_EQ(dist_param_instance4.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance4.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance4.significand_rand_bits(), expected_srb);

    // Over Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    unsigned test_srb = expected_srb + 120U;

    uqr_dist_param dist_param_instance5(test_srb);

    ASSERT_EQ(dist_param_instance5.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance5.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance5.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_construct_copy)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-102);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(73);
    const unsigned expected_srb = 3U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist_param dist_param_instance2(dist_param_instance1);

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_construct_move)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-101);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(75);
    const unsigned expected_srb = 2U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist_param dist_param_instance2(std::move(dist_param_instance1));

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_assign_copy)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-112);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(70);
    const unsigned expected_srb = 4U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist_param dist_param_instance2(2U);
    dist_param_instance2 = dist_param_instance1;

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_assign_move)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-102);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(35);
    const unsigned expected_srb = 1U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist_param dist_param_instance2(2U);
    dist_param_instance2 = std::move(dist_param_instance1);

    ASSERT_EQ(dist_param_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, param_equality_compare)
{
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-102);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(35);
    const unsigned expected_srb = 1U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist_param dist_param_instance2(2U);
    uqr_dist_param dist_param_instance3 = dist_param_instance1;

    ASSERT_TRUE (dist_param_instance1 == dist_param_instance1);
    ASSERT_FALSE(dist_param_instance1 == dist_param_instance2);
    ASSERT_TRUE (dist_param_instance1 == dist_param_instance3);
    ASSERT_FALSE(dist_param_instance2 == dist_param_instance1);
    ASSERT_TRUE (dist_param_instance2 == dist_param_instance2);
    ASSERT_FALSE(dist_param_instance2 == dist_param_instance3);
    ASSERT_TRUE (dist_param_instance3 == dist_param_instance1);
    ASSERT_FALSE(dist_param_instance3 == dist_param_instance2);
    ASSERT_TRUE (dist_param_instance3 == dist_param_instance3);

    ASSERT_FALSE(dist_param_instance1 != dist_param_instance1);
    ASSERT_TRUE (dist_param_instance1 != dist_param_instance2);
    ASSERT_FALSE(dist_param_instance1 != dist_param_instance3);
    ASSERT_TRUE (dist_param_instance2 != dist_param_instance1);
    ASSERT_FALSE(dist_param_instance2 != dist_param_instance2);
    ASSERT_TRUE (dist_param_instance2 != dist_param_instance3);
    ASSERT_FALSE(dist_param_instance3 != dist_param_instance1);
    ASSERT_TRUE (dist_param_instance3 != dist_param_instance2);
    ASSERT_FALSE(dist_param_instance3 != dist_param_instance3);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_default)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(1);
    const unsigned expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist dist_instance1;
    using actual_uqr_dist_rt = typename decltype(dist_instance1)::result_type;

    ASSERT_TRUE((std::is_same<actual_uqr_dist_rt, expected_uqr_dist_rt>::value));
    ASSERT_EQ(dist_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_instance1.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_a_b_srb)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    // Any
    auto expected_a       = static_cast<expected_uqr_dist_rt>(-137);
    auto expected_b       = static_cast<expected_uqr_dist_rt>(271);
    unsigned expected_srb = 2U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_instance1.significand_rand_bits(), expected_srb);

    // Zero
    expected_a   = static_cast<expected_uqr_dist_rt>(47);
    expected_b   = static_cast<expected_uqr_dist_rt>(63);
    expected_srb = 0U;

    uqr_dist dist_instance2(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);

    // Almost Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(-55);
    expected_b   = static_cast<expected_uqr_dist_rt>(-15);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 3U;

    uqr_dist dist_instance3(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance3.a(),                     expected_a);
    ASSERT_EQ(dist_instance3.b(),                     expected_b);
    ASSERT_EQ(dist_instance3.significand_rand_bits(), expected_srb);

    // Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(2);
    expected_b   = static_cast<expected_uqr_dist_rt>(2);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist dist_instance4(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance4.a(),                     expected_a);
    ASSERT_EQ(dist_instance4.b(),                     expected_b);
    ASSERT_EQ(dist_instance4.significand_rand_bits(), expected_srb);

    // Over Maximum
    expected_a   = static_cast<expected_uqr_dist_rt>(-3);
    expected_b   = static_cast<expected_uqr_dist_rt>(0);
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    unsigned test_srb = expected_srb + 1U;

    uqr_dist dist_instance5(expected_a, expected_b, test_srb);

    ASSERT_EQ(dist_instance5.a(),                     expected_a);
    ASSERT_EQ(dist_instance5.b(),                     expected_b);
    ASSERT_EQ(dist_instance5.significand_rand_bits(), expected_srb);

    // Throw std::invalid_argument (a > b)
    expected_a   = static_cast<expected_uqr_dist_rt>(-40);
    expected_b   = static_cast<expected_uqr_dist_rt>(-80);
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist dist_instance6(expected_a, expected_b, test_srb);
    }, std::invalid_argument);

    // Throw std::invalid_argument (a is infinite)
    expected_a   = -std::numeric_limits<expected_uqr_dist_rt>::infinity();
    expected_b   = static_cast<expected_uqr_dist_rt>(-80);
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist dist_instance7(expected_a, expected_b, test_srb);
    }, std::invalid_argument);

    // Throw std::invalid_argument (b is infinite)
    expected_a   = static_cast<expected_uqr_dist_rt>(-40);
    expected_b   = std::numeric_limits<expected_uqr_dist_rt>::infinity();
    expected_srb = 1U;

    ASSERT_THROW({
        uqr_dist dist_instance8(expected_a, expected_b, test_srb);
    }, std::invalid_argument);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_param)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(2);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(17);
    const unsigned expected_srb = 3U;

    uqr_dist_param dist_param_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance1(dist_param_instance1);

    ASSERT_EQ(dist_param_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_param_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_param_instance1.significand_rand_bits(), expected_srb);

    ASSERT_EQ(dist_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_instance1.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_srb)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b = static_cast<expected_uqr_dist_rt>(1);

    // Any
    unsigned expected_srb = 3U;

    uqr_dist dist_instance1(expected_srb);

    ASSERT_EQ(dist_instance1.a(),                     expected_a);
    ASSERT_EQ(dist_instance1.b(),                     expected_b);
    ASSERT_EQ(dist_instance1.significand_rand_bits(), expected_srb);

    // Zero
    expected_srb = 0U;

    uqr_dist dist_instance2(expected_srb);

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);

    // Almost Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 2U;

    uqr_dist dist_instance3(expected_srb);

    ASSERT_EQ(dist_instance3.a(),                     expected_a);
    ASSERT_EQ(dist_instance3.b(),                     expected_b);
    ASSERT_EQ(dist_instance3.significand_rand_bits(), expected_srb);

    // Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    uqr_dist dist_instance4(expected_srb);

    ASSERT_EQ(dist_instance4.a(),                     expected_a);
    ASSERT_EQ(dist_instance4.b(),                     expected_b);
    ASSERT_EQ(dist_instance4.significand_rand_bits(), expected_srb);

    // Over Maximum
    expected_srb = std::numeric_limits<expected_uqr_dist_rt>::digits - 1U;

    unsigned test_srb = expected_srb + 10U;

    uqr_dist dist_instance5(test_srb);

    ASSERT_EQ(dist_instance5.a(),                     expected_a);
    ASSERT_EQ(dist_instance5.b(),                     expected_b);
    ASSERT_EQ(dist_instance5.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_copy)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-122);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(33);
    const unsigned expected_srb = 5U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(dist_instance1);

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, construct_move)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(15);
    const unsigned expected_srb = 1U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(std::move(dist_instance1));

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, assign_copy)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-1);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(1);
    const unsigned expected_srb = 3U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(2U);
    dist_instance2 = dist_instance1;

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, assign_move)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-107);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-36);
    const unsigned expected_srb = 2U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(3U);
    dist_instance2 = std::move(dist_instance1);

    ASSERT_EQ(dist_instance2.a(),                     expected_a);
    ASSERT_EQ(dist_instance2.b(),                     expected_b);
    ASSERT_EQ(dist_instance2.significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, get_param)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-22);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-17);
    const unsigned expected_srb = 2U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance1.param().a(),                     expected_a);
    ASSERT_EQ(dist_instance1.param().b(),                     expected_b);
    ASSERT_EQ(dist_instance1.param().significand_rand_bits(), expected_srb);
}

TYPED_TEST(uniform_quantized_real_distribution_test, set_param)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-122);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-67);
    const unsigned expected_srb = 3U;

    uqr_dist dist_instance_ref(expected_a, expected_b, expected_srb);

    // Custom Parameters
    uqr_dist dist_instance1(1U);
    dist_instance1.param(uqr_dist_param(expected_a, expected_b, expected_srb));

    ASSERT_EQ(dist_instance1.param().a(),                     expected_a);
    ASSERT_EQ(dist_instance1.param().b(),                     expected_b);
    ASSERT_EQ(dist_instance1.param().significand_rand_bits(), expected_srb);

    ASSERT_TRUE(dist_instance1 == dist_instance_ref);

    // From Other Distribution
    uqr_dist dist_instance2(2U);
    dist_instance2.param(dist_instance1.param());

    ASSERT_EQ(dist_instance1.param().a(),                     expected_a);
    ASSERT_EQ(dist_instance1.param().b(),                     expected_b);
    ASSERT_EQ(dist_instance1.param().significand_rand_bits(), expected_srb);

    ASSERT_TRUE(dist_instance2 == dist_instance_ref);
}

TYPED_TEST(uniform_quantized_real_distribution_test, get_member_param_equivalence)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(22);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(27);
    const unsigned expected_srb = 4U;

    // Default Constructor
    uqr_dist_param dist_param_instance1;
    uqr_dist dist_instance1;

    ASSERT_EQ(dist_instance1.a(),                     dist_instance1.param().a());
    ASSERT_EQ(dist_instance1.b(),                     dist_instance1.param().b());
    ASSERT_EQ(dist_instance1.significand_rand_bits(), dist_instance1.param().significand_rand_bits());

    ASSERT_EQ(dist_instance1.a(),                     dist_param_instance1.a());
    ASSERT_EQ(dist_instance1.b(),                     dist_param_instance1.b());
    ASSERT_EQ(dist_instance1.significand_rand_bits(), dist_param_instance1.significand_rand_bits());

    // Constructor (a, b, srb)
    uqr_dist_param dist_param_instance2(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance2.a(),                     dist_instance2.param().a());
    ASSERT_EQ(dist_instance2.b(),                     dist_instance2.param().b());
    ASSERT_EQ(dist_instance2.significand_rand_bits(), dist_instance2.param().significand_rand_bits());

    ASSERT_EQ(dist_instance2.a(),                     dist_param_instance2.a());
    ASSERT_EQ(dist_instance2.b(),                     dist_param_instance2.b());
    ASSERT_EQ(dist_instance2.significand_rand_bits(), dist_param_instance2.significand_rand_bits());

    // Constructor (srb)
    uqr_dist_param dist_param_instance3(expected_srb);
    uqr_dist dist_instance3(expected_srb);

    ASSERT_EQ(dist_instance3.a(),                     dist_instance3.param().a());
    ASSERT_EQ(dist_instance3.b(),                     dist_instance3.param().b());
    ASSERT_EQ(dist_instance3.significand_rand_bits(), dist_instance3.param().significand_rand_bits());

    ASSERT_EQ(dist_instance3.a(),                     dist_param_instance3.a());
    ASSERT_EQ(dist_instance3.b(),                     dist_param_instance3.b());
    ASSERT_EQ(dist_instance3.significand_rand_bits(), dist_param_instance3.significand_rand_bits());
}

TYPED_TEST(uniform_quantized_real_distribution_test, get_min)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-99);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-97);
    const unsigned expected_srb = 3U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance1.min(), expected_a);
}

TYPED_TEST(uniform_quantized_real_distribution_test, get_max)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-99);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-97);
    const unsigned expected_srb = 3U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);

    ASSERT_EQ(dist_instance1.max(), expected_b);
}

TYPED_TEST(uniform_quantized_real_distribution_test, equality_compare)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(102);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(105);
    const unsigned expected_srb = 4U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);
    uqr_dist dist_instance2(2U);
    uqr_dist dist_instance3(dist_instance1);

    ASSERT_TRUE (dist_instance1 == dist_instance1);
    ASSERT_FALSE(dist_instance1 == dist_instance2);
    ASSERT_TRUE (dist_instance1 == dist_instance3);
    ASSERT_FALSE(dist_instance2 == dist_instance1);
    ASSERT_TRUE (dist_instance2 == dist_instance2);
    ASSERT_FALSE(dist_instance2 == dist_instance3);
    ASSERT_TRUE (dist_instance3 == dist_instance1);
    ASSERT_FALSE(dist_instance3 == dist_instance2);
    ASSERT_TRUE (dist_instance3 == dist_instance3);

    ASSERT_FALSE(dist_instance1 != dist_instance1);
    ASSERT_TRUE (dist_instance1 != dist_instance2);
    ASSERT_FALSE(dist_instance1 != dist_instance3);
    ASSERT_TRUE (dist_instance2 != dist_instance1);
    ASSERT_FALSE(dist_instance2 != dist_instance2);
    ASSERT_TRUE (dist_instance2 != dist_instance3);
    ASSERT_FALSE(dist_instance3 != dist_instance1);
    ASSERT_TRUE (dist_instance3 != dist_instance2);
    ASSERT_FALSE(dist_instance3 != dist_instance3);
}

TYPED_TEST(uniform_quantized_real_distribution_test, serialize)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-77);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(17);
    const unsigned expected_srb = 4U;

    uqr_dist dist_instance1(expected_a, expected_b, expected_srb);

    // Preserve Stream Formatting #1
    const auto before_flags1 = std::cout.flags();
    const auto before_fill1  = std::cout.fill();
    const auto before_prec1  = std::cout.precision();
    std::cout << dist_instance1 << std::endl;
    const auto after_flags1 = std::cout.flags();
    const auto after_fill1  = std::cout.fill();
    const auto after_prec1  = std::cout.precision();

    ASSERT_FALSE(!std::cout);
    ASSERT_EQ(before_flags1, after_flags1);
    ASSERT_EQ(before_fill1,  after_fill1);
    ASSERT_EQ(before_prec1,  after_prec1);

    // Preserve Stream Formatting #2
    std::wstringstream ss2;
    ss2 << std::oct << std::setprecision(5) << std::boolalpha << std::setfill(ss2.widen('#'));

    const auto before_flags2 = ss2.flags();
    const auto before_fill2  = ss2.fill();
    const auto before_prec2  = ss2.precision();
    ss2 << dist_instance1;
    const auto after_flags2 = ss2.flags();
    const auto after_fill2  = ss2.fill();
    const auto after_prec2  = ss2.precision();

    ASSERT_FALSE(!ss2);
    ASSERT_EQ(before_flags2, after_flags2);
    ASSERT_EQ(before_fill2,  after_fill2);
    ASSERT_EQ(before_prec2,  after_prec2);

    // Preserve Stream Formatting #3
    std::wstringstream ss3;
    ss3 << std::dec << std::setprecision(5) << std::right << std::setw(400) << std::skipws
        << std::noboolalpha << std::setfill(ss3.widen('*'));

    const auto before_flags3 = ss3.flags();
    const auto before_fill3  = ss3.fill();
    const auto before_prec3  = ss3.precision();
    ss3 << dist_instance1;
    const auto after_flags3 = ss3.flags();
    const auto after_fill3  = ss3.fill();
    const auto after_prec3  = ss3.precision();

    ASSERT_FALSE(!ss3);
    ASSERT_EQ(before_flags3, after_flags3);
    ASSERT_EQ(before_fill3,  after_fill3);
    ASSERT_EQ(before_prec3,  after_prec3);

    // Serialize Do Not Change Internal State.
    std::wstringstream ss4;
    ss4 << std::oct << std::setprecision(5) << std::boolalpha << std::setfill(ss4.widen('#'));

    ss4 << dist_instance1;

    ASSERT_FALSE(!ss4);
    ASSERT_EQ(ss2.str(), ss4.str());
}

TYPED_TEST(uniform_quantized_real_distribution_test, deserialize)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(82);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(99);
    const unsigned expected_srb = 5U;

    uqr_dist dist_instance_ref(expected_a, expected_b, expected_srb);

    uqr_dist dist_instance_base(dist_instance_ref);

    // Valid Deserialization (Narrow String)
    std::stringstream ss1, ss1_1, ss1_2;
    uqr_dist dist_instance1, dist_instance2;

    ss1 << "    " << dist_instance_base << dist_instance_base;
    ss1.seekg(0, std::ios::beg);
    const auto before_flags1 = ss1.flags();
    ss1 >> dist_instance1 >> dist_instance2;
    const auto after_flags1  = ss1.flags();

    ASSERT_FALSE(!ss1);
    ASSERT_EQ(before_flags1, after_flags1);

    ASSERT_TRUE(dist_instance1 == dist_instance_ref);
    ASSERT_TRUE(dist_instance2 == dist_instance_ref);

    ss1_1 << dist_instance1;
    ss1_2 << dist_instance2;

    ASSERT_TRUE(dist_instance1 == dist_instance_ref);
    ASSERT_TRUE(dist_instance2 == dist_instance_ref);

    ASSERT_EQ(ss1_1.str(), ss1_2.str());

    // Valid Deserialization (Wide String)
    std::wstringstream ss2, ss2_1, ss2_2;
    uqr_dist dist_instance3, dist_instance4;

    ss2 << L"    " << dist_instance_base << L"  " << dist_instance_base;
    ss2.seekg(0, std::ios::beg);
    const auto before_flags2 = ss2.flags();
    ss2 >> dist_instance3 >> dist_instance4;
    const auto after_flags2  = ss2.flags();

    ASSERT_FALSE(!ss2);
    ASSERT_EQ(before_flags2, after_flags2);

    ASSERT_TRUE(dist_instance3 == dist_instance_ref);
    ASSERT_TRUE(dist_instance4 == dist_instance_ref);

    ss2_1 << dist_instance3;
    ss2_2 << dist_instance4;

    ASSERT_TRUE(dist_instance1 == dist_instance_ref);
    ASSERT_TRUE(dist_instance2 == dist_instance_ref);

    ASSERT_EQ(ss2_1.str(), ss2_2.str());

    // Invalid Deserialization
    std::wstringstream ss3;
    uqr_dist dist_instance5(dist_instance_ref);

    ss3 << L" { A: sjjdaskdjjshdffhjkskkkdk, }";
    ss3.seekg(0, std::ios::beg);
    const auto before_flags3 = ss3.flags();
    ss3 >> dist_instance5;
    const auto after_flags3  = ss3.flags();

    ASSERT_TRUE(ss3.fail());
    ASSERT_FALSE(ss3.bad());
    ASSERT_EQ(before_flags3, after_flags3);

    ASSERT_TRUE(dist_instance5 == dist_instance_ref);
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    constexpr auto val_zero = static_cast<expected_uqr_dist_rt>(0);

    const auto expected_fract   = static_cast<expected_uqr_dist_rt>(0.5);
    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-100);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-92);
    const unsigned expected_srb = 4U;

    std::mt19937_64 g1;
    uqr_dist dist1(expected_a, expected_b, expected_srb);

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        ASSERT_GE(rnd_val, expected_a);
        ASSERT_LE(rnd_val, expected_b);

        expected_uqr_dist_rt actual_ipart;
        ASSERT_EQ(std::modf(rnd_val / expected_fract, &actual_ipart), val_zero);
    }
}

TYPED_TEST(uniform_quantized_real_distribution_test, generate_random_degen_a_b)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(110);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(110);
    const unsigned expected_srb = 3U;

    std::mt19937_64 g1;
    uqr_dist dist1(expected_a, expected_b, expected_srb);

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        ASSERT_EQ(rnd_val, expected_a);
    }
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random_degen_srb)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-100);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(-92);
    const unsigned expected_srb = 0U;

    std::mt19937_64 g1;
    uqr_dist dist1(expected_a, expected_b, expected_srb);

    std::size_t count_a = 0, count_b = 0;
    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        if (rnd_val == expected_a) { ++count_a; }
        if (rnd_val == expected_b) { ++count_b; }

        ASSERT_TRUE((rnd_val == expected_a) || (rnd_val == expected_b));
    }
    std::cout << "a: " << count_a << ", b: " << count_b << std::endl;
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random_c9)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(-2);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(2);
    const unsigned expected_srb = 3U;

    std::mt19937_64 g1;
    uqr_dist dist1(expected_a, expected_b, expected_srb);

    // Using map (no guarantee that all real types will have std::hash implemented).
    std::map<expected_uqr_dist_rt, std::size_t> counts;
    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        ++counts[rnd_val];

        ASSERT_GE(rnd_val, expected_a);
        ASSERT_LE(rnd_val, expected_b);
    }
    ASSERT_LE(counts.size(), 9U); // 2 ^ expected_srb + 1

    std::cout << "elems: " << counts.size();
    for (const auto& count : counts)
    {
        std::cout << ", (" << count.first << ") -> " << count.second;
    }
    std::cout << std::endl;
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random_c17)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(0);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(4);
    const unsigned expected_srb = 4U;

    std::mt19937_64 g1;
    uqr_dist dist1(expected_a, expected_b, expected_srb);

    // Using map (no guarantee that all real types will have std::hash implemented).
    std::map<expected_uqr_dist_rt, std::size_t> counts;
    for (std::size_t ii = 0; ii < 256 * TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        ++counts[rnd_val];

        ASSERT_GE(rnd_val, expected_a);
        ASSERT_LE(rnd_val, expected_b);
    }
    ASSERT_LE(counts.size(), 17U); // 2 ^ expected_srb + 1

    std::cout << "elems: " << counts.size();
    for (const auto& count : counts)
    {
        std::cout << ", (" << count.first << ") -> " << count.second;
    }
    std::cout << std::endl;
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random_param)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    constexpr auto val_zero = static_cast<expected_uqr_dist_rt>(0);

    auto expected_fract   = static_cast<expected_uqr_dist_rt>(0.125);
    auto expected_a       = static_cast<expected_uqr_dist_rt>(0);
    auto expected_b       = static_cast<expected_uqr_dist_rt>(1);
    unsigned expected_srb = 3U;

    const auto test_fract   = static_cast<expected_uqr_dist_rt>(0.5);
    const auto test_a       = static_cast<expected_uqr_dist_rt>(2);
    const auto test_b       = static_cast<expected_uqr_dist_rt>(18);
    const unsigned test_srb = 5U;

    // Temporary Switch Of Param
    std::mt19937_64 g1;
    uqr_dist dist1(test_a, test_b, test_srb);
    uqr_dist_param dist_param1(expected_srb);

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1, dist_param1);

        ASSERT_GE(rnd_val, expected_a);
        ASSERT_LE(rnd_val, expected_b);

        expected_uqr_dist_rt actual_ipart;
        ASSERT_EQ(std::modf(rnd_val / expected_fract, &actual_ipart), val_zero);
    }

    // Original Param
    expected_fract = test_fract;
    expected_a     = test_a;
    expected_b     = test_b;
    expected_srb   = test_srb;

    ASSERT_EQ(dist1.a(),                     expected_a);
    ASSERT_EQ(dist1.b(),                     expected_b);
    ASSERT_EQ(dist1.significand_rand_bits(), expected_srb);

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val = dist1(g1);

        ASSERT_GE(rnd_val, expected_a);
        ASSERT_LE(rnd_val, expected_b);

        expected_uqr_dist_rt actual_ipart;
        ASSERT_EQ(std::modf(rnd_val / expected_fract, &actual_ipart), val_zero);
    }
}

TYPED_TEST(uniform_quantized_real_distribution_test, DISABLED_generate_random_equivalence)
{
    using uqr_dist             = typename TestFixture::uqr_dist;
    using uqr_dist_param       = typename TestFixture::uqr_dist_param;
    using expected_uqr_dist_rt = typename TestFixture::expected_uqr_dist_rt;

    const auto expected_a       = static_cast<expected_uqr_dist_rt>(16);
    const auto expected_b       = static_cast<expected_uqr_dist_rt>(32);
    const unsigned expected_srb = 5U;

    // Equivalent Initialization.
    std::mt19937_64 g1, g2, g3, g4, g5, g6, g7, g8, g9;
    uqr_dist dist1(expected_a, expected_b, expected_srb), dist1_1(expected_a, expected_b, expected_srb);
    uqr_dist dist2(dist1);
    uqr_dist dist3(std::move(dist1_1));
    uqr_dist dist4{uqr_dist_param(expected_a, expected_b, expected_srb)};
    uqr_dist dist5(dist1.param());

    uqr_dist dist6(expected_srb), dist7(expected_srb), dist8(expected_srb), dist9(expected_srb);

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val1 = dist1(g1);
        expected_uqr_dist_rt rnd_val2 = dist2(g2);
        expected_uqr_dist_rt rnd_val3 = dist3(g3);
        expected_uqr_dist_rt rnd_val4 = dist4(g4);
        expected_uqr_dist_rt rnd_val5 = dist5(g5);

        dist6(g6);
        dist7(g7);
        dist8(g8);
        dist9(g9);

        ASSERT_EQ(rnd_val1, rnd_val2);
        ASSERT_EQ(rnd_val1, rnd_val3);
        ASSERT_EQ(rnd_val1, rnd_val4);
        ASSERT_EQ(rnd_val1, rnd_val5);
    }

    // Equivalent Assignment And Serialization.
    dist6.reset();
    dist6 = dist1;
    uqr_dist dist2_1(dist2);
    dist7 = std::move(dist2_1);
    std::stringstream ss;
    ss << dist3 << " " << dist4;
    dist8.reset();
    ss >> dist8 >> dist9;

    for (std::size_t ii = 0; ii < TestFixture::rnd_iter_num; ++ii)
    {
        expected_uqr_dist_rt rnd_val1 = dist1(g1);

        dist2(g2);
        dist3(g3);
        dist4(g4);
        dist5(g5);

        expected_uqr_dist_rt rnd_val6 = dist6(g6);
        expected_uqr_dist_rt rnd_val7 = dist7(g7);
        expected_uqr_dist_rt rnd_val8 = dist8(g8);
        expected_uqr_dist_rt rnd_val9 = dist9(g9);

        ASSERT_EQ(rnd_val1, rnd_val6);
        ASSERT_EQ(rnd_val1, rnd_val7);
        ASSERT_EQ(rnd_val1, rnd_val8);
        ASSERT_EQ(rnd_val1, rnd_val9);
    }
}

}  // namespace tests
