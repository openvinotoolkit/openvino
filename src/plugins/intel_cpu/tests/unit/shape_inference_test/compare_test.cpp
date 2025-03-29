// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "compare.hpp"

using namespace ov::cmp;

TEST(safe_compare_test, inputs_signed_64_bits) {
    int64_t a = 2, b = -3;

    EXPECT_FALSE(lt(a, b));
    EXPECT_FALSE(le(a, b));
    EXPECT_TRUE(ge(a, b));
    EXPECT_TRUE(gt(a, b));
}

TEST(safe_compare_test, inputs_signed_32_bits) {
    int32_t a = -1, b = -1;

    EXPECT_FALSE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_TRUE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}

TEST(safe_compare_test, inputs_signed_mixed_bit_lengths) {
    int32_t a = -256;
    int8_t b = 1;

    EXPECT_TRUE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_FALSE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}

TEST(safe_compare_test, a_signed_b_unsigned_32_bits) {
    int32_t a = -256;
    uint32_t b = 1;

    EXPECT_TRUE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_FALSE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}

TEST(safe_compare_test, a_unsigned_b_signed_64_bits) {
    uint64_t a = 256;
    int64_t b = 1000;

    EXPECT_TRUE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_FALSE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}

TEST(safe_compare_test, a_float_b_signed) {
    float a = -256.0;
    int64_t b = -256;

    EXPECT_FALSE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_TRUE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}

TEST(safe_compare_test, a_float_b_unsigned) {
    float a = -256.0;
    uint64_t b = 257;

    EXPECT_TRUE(lt(a, b));
    EXPECT_TRUE(le(a, b));
    EXPECT_FALSE(ge(a, b));
    EXPECT_FALSE(gt(a, b));
}
