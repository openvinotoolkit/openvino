// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/rms_norm.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, rms_norm_v14_attr_comp_type_default) {
    using ov::op::internal::RMSNorm;
    NodeBuilder::opset().insert<RMSNorm>();

    const auto data = std::make_shared<Parameter>(ov::element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(ov::element::i32, PartialShape{1});
    const auto eps = 1e-5f;

    const auto op = std::make_shared<RMSNorm>(data, axes, eps);

    NodeBuilder builder(op, {data, axes});
    auto g_op = ov::as_type_ptr<RMSNorm>(builder.create());

    EXPECT_EQ(g_op->get_compute_type(), op->get_compute_type());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, rms_norm_v14_attr_comp_type_custom) {
    using ov::op::internal::RMSNorm;
    NodeBuilder::opset().insert<RMSNorm>();

    const auto data = std::make_shared<Parameter>(ov::element::f16, PartialShape{2, 3, 8, 6});
    const auto axes = std::make_shared<Parameter>(ov::element::i32, PartialShape{1});
    const auto eps = 1e-5f;
    const auto compute_type = ov::element::f32;

    const auto op = std::make_shared<RMSNorm>(data, axes, eps, compute_type);

    NodeBuilder builder(op, {data, axes});
    auto g_op = ov::as_type_ptr<RMSNorm>(builder.create());

    EXPECT_EQ(g_op->get_compute_type(), op->get_compute_type());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
