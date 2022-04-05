// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, cum_sum_op_default_attributes_no_axis_input) {
    NodeBuilder::get_ops().register_factory<opset3::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto cs = make_shared<op::CumSum>(A);

    NodeBuilder builder(cs);
    auto g_cs = ov::as_type_ptr<opset3::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, cum_sum_op_default_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    auto cs = make_shared<op::CumSum>(A, axis);

    NodeBuilder builder(cs);
    auto g_cs = ov::as_type_ptr<opset3::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}

TEST(attributes, cum_sum_op_custom_attributes) {
    NodeBuilder::get_ops().register_factory<opset3::CumSum>();

    Shape shape{1, 4};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    bool exclusive = true;
    bool reverse = true;
    auto cs = make_shared<op::CumSum>(A, axis, exclusive, reverse);

    NodeBuilder builder(cs);
    auto g_cs = ov::as_type_ptr<opset3::CumSum>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_cs->is_exclusive(), cs->is_exclusive());
    EXPECT_EQ(g_cs->is_reverse(), cs->is_reverse());
}
