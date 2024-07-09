// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <gtest/gtest.h>

#include <memory>

TEST(shape, test_symbol_add) {
    auto A = std::make_shared<ov::Symbol>();
    auto B = std::make_shared<ov::Symbol>();

    ASSERT_TRUE(ov::symbol::are_equal(A + B, B + A));

    auto C = std::make_shared<ov::Symbol>();
    auto D = std::make_shared<ov::Symbol>();

    auto E = A + B;
    auto F = C + D;

    ASSERT_FALSE(ov::symbol::are_equal(E, F));

    ov::symbol::set_equal(A, C);
    ov::symbol::set_equal(B, D);

    ASSERT_TRUE(ov::symbol::are_equal(E, F));
}

TEST(shape, test_symbol_sub) {
    auto A = std::make_shared<ov::Symbol>();
    auto B = std::make_shared<ov::Symbol>();

    ASSERT_FALSE(ov::symbol::are_equal(A - B, B - A));

    auto C = std::make_shared<ov::Symbol>();
    auto D = std::make_shared<ov::Symbol>();

    auto E = A - B;
    auto F = C - D;

    ASSERT_FALSE(ov::symbol::are_equal(E, F));

    ov::symbol::set_equal(A, C);
    ov::symbol::set_equal(B, D);

    ASSERT_TRUE(ov::symbol::are_equal(E, F));
}

TEST(shape, test_symbol_add_sub) {
    auto A = std::make_shared<ov::Symbol>();
    auto B = std::make_shared<ov::Symbol>();
    auto C = std::make_shared<ov::Symbol>();

    auto D = A + B;
    auto E = D + C;  // E = A + B + C
    auto F = B + C;

    auto G = E - F;  // G = A + B + C - B - C = A
    ASSERT_TRUE(ov::symbol::are_equal(A, G));
}