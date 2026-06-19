// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_transformations/moe_transformation_utils.hpp"

#include <gtest/gtest.h>

#include "openvino/op/ops.hpp"

/*
 * Unit tests for ov::npuw::moe_utils::is_constant_derived()
 *
 * Covered cases:
 *   Positive:
 *     - Bare Constant node
 *     - Convert(Constant)
 *     - Multiply(Constant, Constant)
 *     - Convert(Multiply(Constant, Constant))   -- nested chain
 *   Negative:
 *     - nullptr input
 *     - Parameter (data-dependent)
 *     - Convert(Parameter)
 *     - Multiply(Constant, Parameter)
 *     - Multiply(Parameter, Constant)            -- symmetric check
 *     - Add(Constant, Constant)                  -- unrecognized op type
 */

namespace {

using ov::npuw::moe_utils::is_constant_derived;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::shared_ptr<ov::op::v0::Constant> make_const_f32(ov::Shape shape) {
    std::vector<float> data(ov::shape_size(shape), 1.0f);
    return ov::op::v0::Constant::create(ov::element::f32, shape, data);
}

static std::shared_ptr<ov::op::v0::Parameter> make_param_f32(ov::PartialShape shape) {
    return std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
}

// ── Positive cases ────────────────────────────────────────────────────────────

TEST(IsconstantDerivedTest, BareConstant) {
    auto c = make_const_f32({4, 16});
    EXPECT_TRUE(is_constant_derived(c));
}

TEST(IsconstantDerivedTest, ConvertOfConstant) {
    auto c = make_const_f32({4, 16});
    auto conv = std::make_shared<ov::op::v0::Convert>(c, ov::element::f16);
    EXPECT_TRUE(is_constant_derived(conv));
}

TEST(IsconstantDerivedTest, MultiplyTwoConstants) {
    auto c0 = make_const_f32({4, 16});
    auto c1 = make_const_f32({4, 1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(c0, c1);
    EXPECT_TRUE(is_constant_derived(mul));
}

TEST(IsconstantDerivedTest, NestedConvertMultiplyConstants) {
    // Convert(Multiply(Constant, Constant)) — typical NF4 weight dequant chain
    auto c0 = make_const_f32({4, 16});
    auto c1 = make_const_f32({4, 1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(c0, c1);
    auto conv = std::make_shared<ov::op::v0::Convert>(mul, ov::element::f16);
    EXPECT_TRUE(is_constant_derived(conv));
}

// ── Negative cases ────────────────────────────────────────────────────────────

TEST(IsconstantDerivedTest, NullInput) {
    EXPECT_FALSE(is_constant_derived(nullptr));
}

TEST(IsconstantDerivedTest, BareParameter) {
    auto p = make_param_f32({1, 16});
    EXPECT_FALSE(is_constant_derived(p));
}

TEST(IsconstantDerivedTest, ConvertOfParameter) {
    auto p = make_param_f32({1, 16});
    auto conv = std::make_shared<ov::op::v0::Convert>(p, ov::element::f16);
    EXPECT_FALSE(is_constant_derived(conv));
}

TEST(IsconstantDerivedTest, MultiplyConstantAndParameter) {
    auto c = make_const_f32({4, 1});
    auto p = make_param_f32({4, 16});
    auto mul = std::make_shared<ov::op::v1::Multiply>(c, p);
    EXPECT_FALSE(is_constant_derived(mul));
}

TEST(IsconstantDerivedTest, MultiplyParameterAndConstant) {
    // Symmetric: swapped operand order must also return false
    auto p = make_param_f32({4, 16});
    auto c = make_const_f32({4, 1});
    auto mul = std::make_shared<ov::op::v1::Multiply>(p, c);
    EXPECT_FALSE(is_constant_derived(mul));
}

TEST(IsconstantDerivedTest, UnrecognizedOpType) {
    // Add is not handled by is_constant_derived — must return false
    auto c0 = make_const_f32({4});
    auto c1 = make_const_f32({4});
    auto add = std::make_shared<ov::op::v1::Add>(c0, c1);
    EXPECT_FALSE(is_constant_derived(add));
}

}  // namespace
