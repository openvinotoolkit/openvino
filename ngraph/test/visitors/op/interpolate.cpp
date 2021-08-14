// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/interpolate.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, interpolate_op) {
    NodeBuilder::get_ops().register_factory<op::v0::Interpolate>();
    auto img = make_shared<op::Parameter>(element::f32, Shape{1, 3, 32, 32});
    auto out_shape = make_shared<op::Parameter>(element::i32, Shape{2});

    op::v0::InterpolateAttrs interp_atrs;
    interp_atrs.axes = AxisSet{1, 2};
    interp_atrs.mode = "cubic";
    interp_atrs.align_corners = true;
    interp_atrs.antialias = true;
    interp_atrs.pads_begin = vector<size_t>{0, 0};
    interp_atrs.pads_end = vector<size_t>{0, 0};

    auto interpolate = make_shared<op::v0::Interpolate>(img, out_shape, interp_atrs);
    NodeBuilder builder(interpolate);
    auto g_interpolate = as_type_ptr<op::v0::Interpolate>(builder.create());

    const auto i_attrs = interpolate->get_attrs();
    const auto g_i_attrs = g_interpolate->get_attrs();

    EXPECT_EQ(g_i_attrs.axes, i_attrs.axes);
    EXPECT_EQ(g_i_attrs.mode, i_attrs.mode);
    EXPECT_EQ(g_i_attrs.align_corners, i_attrs.align_corners);
    EXPECT_EQ(g_i_attrs.antialias, i_attrs.antialias);
    EXPECT_EQ(g_i_attrs.pads_begin, i_attrs.pads_begin);
    EXPECT_EQ(g_i_attrs.pads_end, i_attrs.pads_end);
}
