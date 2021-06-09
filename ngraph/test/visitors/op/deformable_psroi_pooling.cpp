// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, deformable_psroi_pooling_op)
{
    NodeBuilder::get_ops().register_factory<opset1::DeformablePSROIPooling>();
    auto input = make_shared<op::Parameter>(element::f32, Shape{2, 16, 67, 32});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});

    const int output_dim = 4;
    const float spatial_scale = 0.0625;
    const int group_size = 2;
    string mode = "bilinear_deformable";
    const int spatial_bins_x = 2;
    const int spatial_bins_y = 3;
    const float trans_std = 0.1;
    const int part_size = 3;

    auto op = make_shared<opset1::DeformablePSROIPooling>(
        input, coords, output_dim, spatial_scale, group_size, mode, spatial_bins_x, spatial_bins_y, trans_std, part_size);
    NodeBuilder builder(op);
    auto g_op = as_type_ptr<opset1::DeformablePSROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_dim(), op->get_output_dim());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_group_size(), op->get_group_size());
    EXPECT_EQ(g_op->get_mode(), op->get_mode());
    EXPECT_EQ(g_op->get_spatial_bins_x(), op->get_spatial_bins_x());
    EXPECT_EQ(g_op->get_spatial_bins_y(), op->get_spatial_bins_y());
    EXPECT_EQ(g_op->get_trans_std(), op->get_trans_std());
    EXPECT_EQ(g_op->get_part_size(), op->get_part_size());
}
