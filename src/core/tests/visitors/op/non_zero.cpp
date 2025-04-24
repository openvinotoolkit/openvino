// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_zero.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, non_zero_op_default) {
    NodeBuilder::opset().insert<ov::op::v3::NonZero>();
    const auto data_node = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<ov::op::v3::NonZero>(data_node);

    NodeBuilder builder(non_zero, {data_node});
    EXPECT_NO_THROW(auto g_non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(builder.create()));

    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i64);
}

TEST(attributes, non_zero_op_i32) {
    NodeBuilder::opset().insert<ov::op::v3::NonZero>();
    const auto data_node = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<ov::op::v3::NonZero>(data_node, element::i32);

    NodeBuilder builder(non_zero, {data_node});
    EXPECT_NO_THROW(auto g_non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(builder.create()));
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i32);
}

TEST(attributes, non_zero_op_i32_string) {
    NodeBuilder::opset().insert<ov::op::v3::NonZero>();
    const auto data_node = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<ov::op::v3::NonZero>(data_node, "i32");

    NodeBuilder builder(non_zero, {data_node});
    EXPECT_NO_THROW(auto g_non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(builder.create()));
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i32);
}

TEST(attributes, non_zero_op_i64) {
    NodeBuilder::opset().insert<ov::op::v3::NonZero>();
    const auto data_node = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<ov::op::v3::NonZero>(data_node, element::i64);

    NodeBuilder builder(non_zero, {data_node});
    EXPECT_NO_THROW(auto g_non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(builder.create()));
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i64);
}

TEST(attributes, non_zero_op_i64_string) {
    NodeBuilder::opset().insert<ov::op::v3::NonZero>();
    const auto data_node = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    const auto non_zero = make_shared<ov::op::v3::NonZero>(data_node, "i64");

    NodeBuilder builder(non_zero, {data_node});
    EXPECT_NO_THROW(auto g_non_zero = ov::as_type_ptr<ov::op::v3::NonZero>(builder.create()));
    const auto expected_attr_count = 1;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(non_zero->get_output_type(), element::i64);
}
