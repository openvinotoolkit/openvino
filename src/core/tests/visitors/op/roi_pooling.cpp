// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roi_pooling.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, roi_pooling_op) {
    NodeBuilder::opset().insert<ov::op::v0::ROIPooling>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4, 5});
    const auto coords = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 5});

    const auto op = make_shared<ov::op::v0::ROIPooling>(data, coords, Shape{5, 5}, 0.123f, "bilinear");
    NodeBuilder builder(op, {data, coords});
    const auto g_op = ov::as_type_ptr<ov::op::v0::ROIPooling>(builder.create());

    EXPECT_EQ(g_op->get_output_roi(), op->get_output_roi());
    EXPECT_EQ(g_op->get_spatial_scale(), op->get_spatial_scale());
    EXPECT_EQ(g_op->get_method(), op->get_method());
}
