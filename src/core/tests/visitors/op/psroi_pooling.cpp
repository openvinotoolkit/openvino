// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, psroi_pooling_op) {
    NodeBuilder::get_ops().register_factory<opset1::PSROIPooling>();
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});

    const int64_t output_dim = 64;
    const int64_t group_size = 4;
    const float spatial_scale = 0.0625;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    string mode = "average";

    auto psroi_pool = make_shared<opset1::PSROIPooling>(input,
                                                        coords,
                                                        output_dim,
                                                        group_size,
                                                        spatial_scale,
                                                        spatial_bins_x,
                                                        spatial_bins_y,
                                                        mode);
    NodeBuilder builder(psroi_pool);
    auto g_psroi_pool = ov::as_type_ptr<opset1::PSROIPooling>(builder.create());

    EXPECT_EQ(g_psroi_pool->get_output_dim(), psroi_pool->get_output_dim());
    EXPECT_EQ(g_psroi_pool->get_group_size(), psroi_pool->get_group_size());
    EXPECT_EQ(g_psroi_pool->get_spatial_scale(), psroi_pool->get_spatial_scale());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_x(), psroi_pool->get_spatial_bins_x());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_y(), psroi_pool->get_spatial_bins_y());
    EXPECT_EQ(g_psroi_pool->get_mode(), psroi_pool->get_mode());
}
