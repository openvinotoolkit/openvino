// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grouped_matmul.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::test::NodeBuilder;

TEST(attributes, grouped_matmul_v17_3d_3d) {
    NodeBuilder::opset().insert<ov::op::v17::GroupedMatMul>();
    const auto mat_a = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{3, 4, 64});
    const auto mat_b = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{3, 64, 128});

    const auto op = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b);
    NodeBuilder builder(op, {mat_a, mat_b});
    auto g_op = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(builder.create());

    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
}

TEST(attributes, grouped_matmul_v17_2d_3d_with_offsets) {
    NodeBuilder::opset().insert<ov::op::v17::GroupedMatMul>();
    const auto mat_a = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{16, 64});
    const auto mat_b = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{3, 64, 128});
    const auto offsets = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{3});

    const auto op = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b, offsets);
    NodeBuilder builder(op, {mat_a, mat_b, offsets});
    auto g_op = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(builder.create());

    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
}

TEST(attributes, grouped_matmul_v17_2d_2d_with_offsets) {
    NodeBuilder::opset().insert<ov::op::v17::GroupedMatMul>();
    const auto mat_a = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{64, 16});
    const auto mat_b = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{16, 128});
    const auto offsets = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{3});

    const auto op = std::make_shared<ov::op::v17::GroupedMatMul>(mat_a, mat_b, offsets);
    NodeBuilder builder(op, {mat_a, mat_b, offsets});
    auto g_op = ov::as_type_ptr<ov::op::v17::GroupedMatMul>(builder.create());

    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
}
}  // namespace ov::test
