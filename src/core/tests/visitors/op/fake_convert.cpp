// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_convert.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::Shape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, fake_convert_v13_attributes_default) {
    using ov::op::v13::FakeConvert;
    NodeBuilder::opset().insert<FakeConvert>();
    const auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{});
    const auto shift = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{});

    const auto op = std::make_shared<FakeConvert>(data, scale, shift);

    NodeBuilder builder(op, {data, scale, shift});
    auto g_op = ov::as_type_ptr<FakeConvert>(builder.create());

    EXPECT_EQ(g_op->get_destination_type(), op->get_destination_type());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, fake_convert_v13_attributes_custom) {
    using ov::op::v13::FakeConvert;
    NodeBuilder::opset().insert<FakeConvert>();
    const auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 3, 8, 6});
    const auto scale = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{});
    const auto shift = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{});

    const auto op = std::make_shared<FakeConvert>(data, scale, shift, "f8e5m2");

    NodeBuilder builder(op, {data, scale, shift});
    auto g_op = ov::as_type_ptr<FakeConvert>(builder.create());

    EXPECT_EQ(g_op->get_destination_type(), op->get_destination_type());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
