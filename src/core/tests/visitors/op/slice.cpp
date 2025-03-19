// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/slice.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, slice_op_no_axes) {
    NodeBuilder::opset().insert<ov::op::v8::Slice>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 5, 4});
    const auto start = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    const auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    const auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});

    const auto op = make_shared<ov::op::v8::Slice>(data, start, stop, step);
    NodeBuilder builder(op, {data, start, stop, step});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v8::Slice>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(attributes, slice_op_with_axes) {
    NodeBuilder::opset().insert<ov::op::v8::Slice>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 5, 4});
    const auto start = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    const auto stop = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    const auto step = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});
    const auto axes = make_shared<ov::op::v0::Parameter>(element::i32, Shape{4});

    const auto op = make_shared<ov::op::v8::Slice>(data, start, stop, step, axes);
    NodeBuilder builder(op, {data, start, stop, step, axes});
    EXPECT_NO_THROW(auto g_op = ov::as_type_ptr<ov::op::v8::Slice>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
