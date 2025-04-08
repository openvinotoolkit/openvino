// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/eye.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, eye_op) {
    NodeBuilder::opset().insert<op::v9::Eye>();
    auto num_rows = make_shared<op::v0::Constant>(element::i32, Shape{}, 10);
    auto num_columns = make_shared<op::v0::Constant>(element::i32, Shape{}, 2);
    auto diagonal_index = make_shared<op::v0::Constant>(element::i32, Shape{}, 0);

    const auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, element::Type_t::u8);
    NodeBuilder builder(eye, {num_rows, num_columns, diagonal_index});
    auto g_eye = ov::as_type_ptr<op::v9::Eye>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_eye->get_out_type(), eye->get_out_type());
}

TEST(attributes, eye_batch_shape_op) {
    NodeBuilder::opset().insert<op::v9::Eye>();
    auto num_rows = make_shared<op::v0::Constant>(element::i32, Shape{}, 2);
    auto num_columns = make_shared<op::v0::Constant>(element::i32, Shape{}, 5);
    auto diagonal_index = make_shared<op::v0::Constant>(element::i32, Shape{}, 1);
    auto batch_shape = make_shared<op::v0::Constant>(element::i32, Shape{3}, std::vector<int32_t>{1, 2, 3});

    const auto eye = make_shared<op::v9::Eye>(num_rows, num_columns, diagonal_index, batch_shape, element::Type_t::i32);
    NodeBuilder builder(eye, {num_rows, num_columns, diagonal_index, batch_shape});
    auto g_eye = ov::as_type_ptr<op::v9::Eye>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_eye->get_out_type(), eye->get_out_type());
}
