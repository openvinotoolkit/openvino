// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace std;
using namespace ov::symbol;

TEST(shape, test_symbol_sum) {
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
