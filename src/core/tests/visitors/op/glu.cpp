// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/glu.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::op::internal::GLU;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, glu_attr_Swish) {
    NodeBuilder::opset().insert<GLU>();

    int64_t axis = -1;
    int64_t split_lenghts = 3;
    auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 1, 6});
    auto op = std::make_shared<GLU>(data, axis, split_lenghts, GLU::GluType::Swish, 0);

    NodeBuilder builder(op, {data});
    auto g_op = ov::as_type_ptr<GLU>(builder.create());

    EXPECT_EQ(g_op->get_axis(), op->get_axis());
    EXPECT_EQ(g_op->get_split_lengths(), op->get_split_lengths());
    EXPECT_EQ(g_op->get_glu_type(), op->get_glu_type());
    EXPECT_EQ(g_op->get_split_to_glu_idx(), op->get_split_to_glu_idx());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, glu_attr_Gelu) {
    NodeBuilder::opset().insert<GLU>();

    int64_t axis = 2;
    int64_t split_lenghts = 3;
    auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
    auto op = std::make_shared<GLU>(data, axis, split_lenghts, GLU::GluType::Gelu, 1, ov::element::f16);

    NodeBuilder builder(op, {data});
    auto g_op = ov::as_type_ptr<GLU>(builder.create());

    EXPECT_EQ(g_op->get_axis(), op->get_axis());
    EXPECT_EQ(g_op->get_split_lengths(), op->get_split_lengths());
    EXPECT_EQ(g_op->get_glu_type(), op->get_glu_type());
    EXPECT_EQ(g_op->get_split_to_glu_idx(), op->get_split_to_glu_idx());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, glu_attr_Gelu_Tanh) {
    NodeBuilder::opset().insert<GLU>();

    int64_t axis = 2;
    int64_t split_lenghts = 3;
    auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
    auto op = std::make_shared<GLU>(data, axis, split_lenghts, GLU::GluType::Gelu_Tanh, 1, ov::element::f16);

    NodeBuilder builder(op, {data});
    auto g_op = ov::as_type_ptr<GLU>(builder.create());

    EXPECT_EQ(g_op->get_axis(), op->get_axis());
    EXPECT_EQ(g_op->get_split_lengths(), op->get_split_lengths());
    EXPECT_EQ(g_op->get_glu_type(), op->get_glu_type());
    EXPECT_EQ(g_op->get_split_to_glu_idx(), op->get_split_to_glu_idx());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
