// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cum_sum.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, cum_sum_op_default_attributes_no_axis_input) {
    NodeBuilder::opset().insert<ov::op::v0::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto cs = make_shared<op::v0::CumSum>(A);

    NodeBuilder builder(cs, {A});
    auto g_cs = ov::as_type_ptr<op::v0::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, cum_sum_op_default_attributes) {
    NodeBuilder::opset().insert<op::v0::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto axis = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    auto cs = make_shared<op::v0::CumSum>(A, axis);

    NodeBuilder builder(cs, {A, axis});
    auto g_cs = ov::as_type_ptr<op::v0::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, cum_sum_op_custom_attributes) {
    NodeBuilder::opset().insert<op::v0::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<ov::op::v0::Parameter>(element::f32, shape);
    auto axis = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1});
    bool exclusive = true;
    bool reverse = true;
    auto cs = make_shared<op::v0::CumSum>(A, axis, exclusive, reverse);

    NodeBuilder builder(cs, {A, axis});
    auto g_cs = ov::as_type_ptr<op::v0::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}
