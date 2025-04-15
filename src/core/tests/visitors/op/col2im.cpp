// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::op::v15::Col2Im;
using ov::test::NodeBuilder;

TEST(attributes, col2im_v15_attr_comp_type_default) {
    NodeBuilder::opset().insert<Col2Im>();

    const auto data = std::make_shared<Parameter>(ov::element::i32, PartialShape{3, 12, 81});
    const auto output_size = std::make_shared<Parameter>(ov::element::i64, ov::Shape{2});
    const auto kernel_size = std::make_shared<Parameter>(ov::element::i64, ov::Shape{2});

    const auto op = std::make_shared<Col2Im>(data, output_size, kernel_size);

    NodeBuilder builder(op, {data, output_size, kernel_size});
    auto g_op = ov::as_type_ptr<Col2Im>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
    EXPECT_EQ(g_op->get_dilations(), op->get_dilations());
    EXPECT_EQ(g_op->get_pads_begin(), op->get_pads_begin());
    EXPECT_EQ(g_op->get_pads_end(), op->get_pads_end());
}

TEST(attributes, col2im_v15_attr_comp_type_custom) {
    NodeBuilder::opset().insert<Col2Im>();

    const auto data = std::make_shared<Parameter>(ov::element::i32, PartialShape{3, 12, 81});
    const auto output_size = std::make_shared<Parameter>(ov::element::i64, ov::Shape{2});
    const auto kernel_size = std::make_shared<Parameter>(ov::element::i64, ov::Shape{2});
    const auto strides = ov::Strides{2, 2};
    const auto dilations = ov::Strides{2, 2};
    const auto pads_begin = ov::Shape{2, 2};
    const auto pads_end = ov::Shape{2, 2};

    const auto op = std::make_shared<Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end);

    NodeBuilder builder(op, {data, output_size, kernel_size});
    auto g_op = ov::as_type_ptr<Col2Im>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
    EXPECT_EQ(g_op->get_dilations(), op->get_dilations());
    EXPECT_EQ(g_op->get_pads_begin(), op->get_pads_begin());
    EXPECT_EQ(g_op->get_pads_end(), op->get_pads_end());
}
