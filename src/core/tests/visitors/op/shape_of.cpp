// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shape_of.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, shapeof_op1) {
    NodeBuilder::opset().insert<op::v0::ShapeOf>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto shapeof = make_shared<op::v0::ShapeOf>(data);
    NodeBuilder builder(shapeof, {data});
    auto g_shapeof = ov::as_type_ptr<op::v0::ShapeOf>(builder.create());

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, shapeof_op3) {
    NodeBuilder::opset().insert<op::v3::ShapeOf>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 4});
    auto shapeof = make_shared<op::v3::ShapeOf>(data, element::Type_t::i64);
    NodeBuilder builder(shapeof, {data});
    auto g_shapeof = ov::as_type_ptr<op::v3::ShapeOf>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
