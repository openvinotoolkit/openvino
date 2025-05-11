// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/swish.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, swish_op) {
    NodeBuilder::opset().insert<ov::op::v4::Swish>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});

    const auto op = make_shared<ov::op::v4::Swish>(data);
    NodeBuilder builder(op, {data});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v4::Swish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, swish_op2) {
    NodeBuilder::opset().insert<ov::op::v4::Swish>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto beta = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    const auto op = make_shared<ov::op::v4::Swish>(data, beta);
    NodeBuilder builder(op, {data, beta});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v4::Swish>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
