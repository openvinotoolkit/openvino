// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/deformable_psroi_pooling.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, deformable_psroi_pooling_op) {
    NodeBuilder::opset().insert<ov::op::v1::DeformablePSROIPooling>();
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 16, 67, 32});
    auto coords = make_shared<ov::op::v0::Parameter>(element::f32, Shape{300, 5});

    const int output_dim = 4;
    const float spatial_scale = 0.0625;
    const int group_size = 2;
    string mode = "bilinear_deformable";
    const int spatial_bins_x = 2;
    const int spatial_bins_y = 3;
    const float trans_std = 0.1f;
    const int part_size = 3;

    auto op = make_shared<ov::op::v1::DeformablePSROIPooling>(input,
                                                              coords,
                                                              output_dim,
                                                              spatial_scale,
                                                              group_size,
                                                              mode,
                                                              spatial_bins_x,
                                                              spatial_bins_y,
                                                              trans_std,
                                                              part_size);
    NodeBuilder builder(op, {input, coords});
    auto g_op = ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_dim(), op->get_output_dim());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_group_size(), op->get_group_size());
    EXPECT_EQ(g_op->get_mode(), op->get_mode());
    EXPECT_EQ(g_op->get_spatial_bins_x(), op->get_spatial_bins_x());
    EXPECT_EQ(g_op->get_spatial_bins_y(), op->get_spatial_bins_y());
    EXPECT_EQ(g_op->get_trans_std(), op->get_trans_std());
    EXPECT_EQ(g_op->get_part_size(), op->get_part_size());
}

TEST(attributes, deformable_psroi_pooling_op2) {
    NodeBuilder::opset().insert<ov::op::v1::DeformablePSROIPooling>();
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 16, 67, 32});
    auto coords = make_shared<ov::op::v0::Parameter>(element::f32, Shape{300, 5});
    auto offset = make_shared<ov::op::v0::Parameter>(element::i64, Shape{300, 2, 2, 2});

    const int output_dim = 4;
    const float spatial_scale = 0.0625;
    const int group_size = 2;
    string mode = "bilinear_deformable";
    const int spatial_bins_x = 2;
    const int spatial_bins_y = 3;
    const float trans_std = 0.1f;
    const int part_size = 3;

    auto op = make_shared<ov::op::v1::DeformablePSROIPooling>(input,
                                                              coords,
                                                              offset,
                                                              output_dim,
                                                              spatial_scale,
                                                              group_size,
                                                              mode,
                                                              spatial_bins_x,
                                                              spatial_bins_y,
                                                              trans_std,
                                                              part_size);
    NodeBuilder builder(op, {input, coords, offset});
    auto g_op = ov::as_type_ptr<ov::op::v1::DeformablePSROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_dim(), op->get_output_dim());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_group_size(), op->get_group_size());
    EXPECT_EQ(g_op->get_mode(), op->get_mode());
    EXPECT_EQ(g_op->get_spatial_bins_x(), op->get_spatial_bins_x());
    EXPECT_EQ(g_op->get_spatial_bins_y(), op->get_spatial_bins_y());
    EXPECT_EQ(g_op->get_trans_std(), op->get_trans_std());
    EXPECT_EQ(g_op->get_part_size(), op->get_part_size());
}
