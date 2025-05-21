// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/sparse_fill_empty_rows.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

TEST(attributes, sparse_fill_empty_rows_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v16::SparseFillEmptyRows>();
    const auto values = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    const auto dense_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{3, 2});
    const auto default_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{});

    const auto sparse_fill_empty_rows =
        std::make_shared<ov::op::v16::SparseFillEmptyRows>(values, dense_shape, indices, default_value);

    ov::test::NodeBuilder builder(sparse_fill_empty_rows, {values, dense_shape, indices, default_value});
    ASSERT_NO_THROW(std::ignore = ov::as_type_ptr<ov::op::v16::SparseFillEmptyRows>(builder.create()));
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
