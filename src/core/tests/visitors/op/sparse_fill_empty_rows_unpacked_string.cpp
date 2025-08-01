// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

TEST(attributes, sparse_fill_empty_rows_unpacked_string_op) {
    ov::test::NodeBuilder::opset().insert<ov::op::v16::SparseFillEmptyRowsUnpackedString>();
    const auto begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6});
    const auto ends = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6});
    const auto symbols = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{35});
    const auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{6, 2});
    const auto dense_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto default_value = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::Shape{5});

    const auto sparse_fill_empty_rows_unpacked_string =
        std::make_shared<ov::op::v16::SparseFillEmptyRowsUnpackedString>(begins,
                                                                         ends,
                                                                         symbols,
                                                                         indices,
                                                                         dense_shape,
                                                                         default_value);

    ov::test::NodeBuilder builder(sparse_fill_empty_rows_unpacked_string,
                                  {begins, ends, symbols, indices, dense_shape, default_value});
    ASSERT_NO_THROW(std::ignore = ov::as_type_ptr<ov::op::v16::SparseFillEmptyRowsUnpackedString>(builder.create()));
    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
