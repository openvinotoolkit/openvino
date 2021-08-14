// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/psroi_pooling.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, psroi_pooling_op) {
    NodeBuilder::get_ops().register_factory<op::v0::PSROIPooling>();
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});

    const int64_t output_dim = 64;
    const int64_t group_size = 4;
    const float spatial_scale = 0.0625;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    string mode = "average";

    auto psroi_pool = make_shared<op::v0::PSROIPooling>(input,
                                                        coords,
                                                        output_dim,
                                                        group_size,
                                                        spatial_scale,
                                                        spatial_bins_x,
                                                        spatial_bins_y,
                                                        mode);
    NodeBuilder builder(psroi_pool);
    auto g_psroi_pool = as_type_ptr<op::v0::PSROIPooling>(builder.create());

    EXPECT_EQ(g_psroi_pool->get_output_dim(), psroi_pool->get_output_dim());
    EXPECT_EQ(g_psroi_pool->get_group_size(), psroi_pool->get_group_size());
    EXPECT_EQ(g_psroi_pool->get_spatial_scale(), psroi_pool->get_spatial_scale());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_x(), psroi_pool->get_spatial_bins_x());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_y(), psroi_pool->get_spatial_bins_y());
    EXPECT_EQ(g_psroi_pool->get_mode(), psroi_pool->get_mode());
}
