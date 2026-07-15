// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "visitors/visitors.hpp"

using ov::op::internal::DynamicQuantize;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

// Whole-dimension group size marker: one scale per whole axis.
constexpr uint64_t WHOLE_DIM = std::numeric_limits<uint64_t>::max();

TEST(attributes, dynamic_quantize_attr_symmetric_per_tensor) {
    NodeBuilder::opset().insert<DynamicQuantize>();

    auto data = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 4, 8});

    DynamicQuantize::Attributes attrs;
    attrs.quantization_type = DynamicQuantize::QuantizationType::Symmetric;
    attrs.quantization_dt = ov::element::u8;
    attrs.scale_dt = ov::element::f32;
    attrs.group_sizes = {WHOLE_DIM, WHOLE_DIM, WHOLE_DIM};
    attrs.output_storage_type = DynamicQuantize::OutputStorageType::Planar;

    auto op = std::make_shared<DynamicQuantize>(data, attrs);

    // NodeBuilder serializes the attributes through visit_attributes and rebuilds the op
    // from them. Without a visit_attributes override the group_sizes are dropped and the
    // rebuild trips the shape_infer rank check (regression guard for the empty-group_sizes bug).
    NodeBuilder builder(op, {data});
    auto g_op = ov::as_type_ptr<DynamicQuantize>(builder.create());

    EXPECT_EQ(g_op->get_quantization_type(), op->get_quantization_type());
    EXPECT_EQ(g_op->get_attrs().quantization_dt, op->get_attrs().quantization_dt);
    EXPECT_EQ(g_op->get_attrs().scale_dt, op->get_attrs().scale_dt);
    EXPECT_EQ(g_op->get_group_sizes(), op->get_group_sizes());
    EXPECT_EQ(g_op->get_scales_zp_output_order(), op->get_scales_zp_output_order());
    EXPECT_EQ(g_op->get_output_storage_type(), op->get_output_storage_type());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_partial_shape(1), op->get_output_partial_shape(1));
}

TEST(attributes, dynamic_quantize_attr_asymmetric_grouped) {
    NodeBuilder::opset().insert<DynamicQuantize>();

    auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{1, 4, 8});

    DynamicQuantize::Attributes attrs;
    attrs.quantization_type = DynamicQuantize::QuantizationType::Asymmetric;
    attrs.quantization_dt = ov::element::i8;
    attrs.scale_dt = ov::element::f16;
    attrs.zp_dt = ov::element::i8;
    attrs.group_sizes = {1, 1, 8};
    attrs.scales_zp_output_order = {0, 1, 2};
    attrs.output_storage_type = DynamicQuantize::OutputStorageType::Planar;

    auto op = std::make_shared<DynamicQuantize>(data, attrs);

    NodeBuilder builder(op, {data});
    auto g_op = ov::as_type_ptr<DynamicQuantize>(builder.create());

    EXPECT_EQ(g_op->get_quantization_type(), op->get_quantization_type());
    EXPECT_EQ(g_op->get_attrs().quantization_dt, op->get_attrs().quantization_dt);
    EXPECT_EQ(g_op->get_attrs().scale_dt, op->get_attrs().scale_dt);
    EXPECT_EQ(g_op->get_attrs().zp_dt, op->get_attrs().zp_dt);
    EXPECT_EQ(g_op->get_group_sizes(), op->get_group_sizes());
    EXPECT_EQ(g_op->get_scales_zp_output_order(), op->get_scales_zp_output_order());
    EXPECT_EQ(g_op->get_output_storage_type(), op->get_output_storage_type());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_partial_shape(1), op->get_output_partial_shape(1));
    EXPECT_EQ(g_op->get_output_partial_shape(2), op->get_output_partial_shape(2));
}
