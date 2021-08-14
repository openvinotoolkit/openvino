// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/roi_pooling.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, roi_pooling_op) {
    NodeBuilder::get_ops().register_factory<op::v0::ROIPooling>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});
    const auto coords = make_shared<op::Parameter>(element::f32, Shape{2, 5});

    const auto op = make_shared<op::v0::ROIPooling>(data, coords, Shape{5, 5}, 0.123, "bilinear");
    NodeBuilder builder(op);
    const auto g_op = as_type_ptr<op::v0::ROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_size(), op->get_output_size());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_method(), op->get_method());
}
