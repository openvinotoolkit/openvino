// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/floor_mod.hpp"

#include "arithmetic_ops.hpp"
#include "openvino/core/bound_evaluation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"

using Type = ::testing::Types<ov::op::v1::FloorMod>;

INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_floormod, ArithmeticOperator, Type);

TEST(type_prop_floormod, constant_bounds) {
    const auto lhs = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {-7, 7, -7, 7});
    const auto rhs = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {3, -3, -3, 3});
    const auto mod = std::make_shared<ov::op::v1::FloorMod>(lhs, rhs);
    const auto& [lower, upper] = ov::util::evaluate_both_bounds(mod);

    ASSERT_TRUE(lower);
    ASSERT_TRUE(upper);
    const std::vector<int64_t> expected{2, -2, -1, 1};
    EXPECT_EQ(std::vector<int64_t>(lower.data<int64_t>(), lower.data<int64_t>() + 4), expected);
    EXPECT_EQ(std::vector<int64_t>(upper.data<int64_t>(), upper.data<int64_t>() + 4), expected);
}

TEST(type_prop_floormod, static_shape_bounds) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{7, 10});
    const auto shape = std::make_shared<ov::op::v3::ShapeOf>(data);
    const auto divisor = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    const auto mod = std::make_shared<ov::op::v1::FloorMod>(shape, divisor);
    const auto bounds = ov::util::evaluate_both_bounds(mod);

    ASSERT_TRUE(bounds.first);
    ASSERT_TRUE(bounds.second);
    EXPECT_EQ(bounds.first.data<int64_t>()[0], 1);
    EXPECT_EQ(bounds.first.data<int64_t>()[1], 1);
    EXPECT_EQ(bounds.second.data<int64_t>()[0], 1);
    EXPECT_EQ(bounds.second.data<int64_t>()[1], 1);
}

TEST(type_prop_floormod, interval_bounds_are_not_evaluated_at_endpoints) {
    for (const auto& shape : {ov::PartialShape{-1}, ov::PartialShape{{0, 7}}, ov::PartialShape{{5, 7}}}) {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        const auto dims = std::make_shared<ov::op::v3::ShapeOf>(data);
        const auto three = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        for (const auto& inputs :
             {ov::OutputVector{dims, dims}, ov::OutputVector{dims, three}, ov::OutputVector{three, dims}}) {
            const auto mod = std::make_shared<ov::op::v1::FloorMod>(inputs[0], inputs[1]);
            const auto bounds = ov::util::evaluate_both_bounds(mod);
            EXPECT_FALSE(bounds.first);
            EXPECT_FALSE(bounds.second);
        }
    }
}
