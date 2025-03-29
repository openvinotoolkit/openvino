// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/psroi_pooling.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, psroi_pooling_op) {
    NodeBuilder::opset().insert<ov::op::v0::PSROIPooling>();
    auto input = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<ov::op::v0::Parameter>(element::f32, Shape{300, 5});

    const int64_t output_dim = 64;
    const int64_t group_size = 4;
    const float spatial_scale = 0.0625f;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    string mode = "average";

    auto psroi_pool = make_shared<ov::op::v0::PSROIPooling>(input,
                                                            coords,
                                                            output_dim,
                                                            group_size,
                                                            spatial_scale,
                                                            spatial_bins_x,
                                                            spatial_bins_y,
                                                            mode);
    NodeBuilder builder(psroi_pool, {input, coords});
    auto g_psroi_pool = ov::as_type_ptr<ov::op::v0::PSROIPooling>(builder.create());

    EXPECT_EQ(g_psroi_pool->get_output_dim(), psroi_pool->get_output_dim());
    EXPECT_EQ(g_psroi_pool->get_group_size(), psroi_pool->get_group_size());
    EXPECT_EQ(g_psroi_pool->get_spatial_scale(), psroi_pool->get_spatial_scale());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_x(), psroi_pool->get_spatial_bins_x());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_y(), psroi_pool->get_spatial_bins_y());
    EXPECT_EQ(g_psroi_pool->get_mode(), psroi_pool->get_mode());
}
