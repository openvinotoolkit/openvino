// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/opsets/opset4.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, swish_op) {
    NodeBuilder::get_ops().register_factory<opset4::Swish>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});

    const auto op = make_shared<opset4::Swish>(data);
    NodeBuilder builder(op, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<opset4::Swish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, swish_op2) {
    NodeBuilder::get_ops().register_factory<opset4::Swish>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto beta = make_shared<op::Parameter>(element::f32, Shape{});

    const auto op = make_shared<opset4::Swish>(data, beta);
    NodeBuilder builder(op, {data, beta});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<opset4::Swish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
