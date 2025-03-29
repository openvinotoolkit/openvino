// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/equal.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/shape_of.hpp"

using ov::op::v0::Convert;
using ov::op::v0::Parameter;
using ov::op::v3::Broadcast;
using ov::op::v3::ShapeOf;

class TypePropEqualV1Test : public TypePropOpTest<ov::op::v1::Equal> {
    // Common test for Equal operator are in type_prop/binary_elementwise.cpp
};

TEST_F(TypePropEqualV1Test, lhs_upper_bound_within_rhs_bounds) {
    constexpr auto et = ov::element::i32;

    const auto lhs = std::make_shared<Parameter>(et, ov::PartialShape{{1, 1}});
    const auto rhs = std::make_shared<Parameter>(et, ov::PartialShape{{0, -1}});
    const auto lhs_shape_of = std::make_shared<ShapeOf>(lhs, et);
    const auto rhs_shape_of = std::make_shared<ShapeOf>(rhs, et);
    const auto op = make_op(lhs_shape_of, rhs_shape_of);

    const auto p = std::make_shared<Parameter>(et, ov::PartialShape{1});
    const auto bc = std::make_shared<Broadcast>(p, std::make_shared<Convert>(op, et));

    EXPECT_EQ(bc->get_output_partial_shape(0), ov::PartialShape({{0, 1}}));
}

TEST_F(TypePropEqualV1Test, rhs_upper_bound_within_lhs_bounds) {
    constexpr auto et = ov::element::i32;

    const auto lhs = std::make_shared<Parameter>(et, ov::PartialShape{{0, -1}});
    const auto rhs = std::make_shared<Parameter>(et, ov::PartialShape{{1, 1}});
    const auto lhs_shape_of = std::make_shared<ShapeOf>(lhs, et);
    const auto rhs_shape_of = std::make_shared<ShapeOf>(rhs, et);
    const auto op = make_op(lhs_shape_of, rhs_shape_of);

    const auto p = std::make_shared<Parameter>(et, ov::PartialShape{1});
    const auto bc = std::make_shared<Broadcast>(p, std::make_shared<Convert>(op, et));

    EXPECT_EQ(bc->get_output_partial_shape(0), ov::PartialShape({{0, 1}}));
}
