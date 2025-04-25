// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, transpose_op) {
    NodeBuilder::opset().insert<ov::op::v1::Transpose>();
    const auto data_input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes_order_input = make_shared<ov::op::v0::Parameter>(element::i32, Shape{3});

    const auto op = make_shared<ov::op::v1::Transpose>(data_input, axes_order_input);

    NodeBuilder builder(op, {data_input, axes_order_input});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v1::Transpose>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
