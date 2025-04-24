// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <gtest/gtest.h>

#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, interpolate_op1) {
    NodeBuilder::opset().insert<ov::op::v0::Interpolate>();
    auto img = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 32, 32});
    auto out_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});

    op::v0::Interpolate::Attributes interp_atrs;
    interp_atrs.axes = AxisSet{1, 2};
    interp_atrs.mode = "cubic";
    interp_atrs.align_corners = true;
    interp_atrs.antialias = true;
    interp_atrs.pads_begin = vector<size_t>{0, 0};
    interp_atrs.pads_end = vector<size_t>{0, 0};

    auto interpolate = make_shared<ov::op::v0::Interpolate>(img, out_shape, interp_atrs);
    NodeBuilder builder(interpolate, {img, out_shape});
    auto g_interpolate = ov::as_type_ptr<ov::op::v0::Interpolate>(builder.create());

    const auto i_attrs = interpolate->get_attrs();
    const auto g_i_attrs = g_interpolate->get_attrs();

    EXPECT_EQ(g_i_attrs.axes, i_attrs.axes);
    EXPECT_EQ(g_i_attrs.mode, i_attrs.mode);
    EXPECT_EQ(g_i_attrs.align_corners, i_attrs.align_corners);
    EXPECT_EQ(g_i_attrs.antialias, i_attrs.antialias);
    EXPECT_EQ(g_i_attrs.pads_begin, i_attrs.pads_begin);
    EXPECT_EQ(g_i_attrs.pads_end, i_attrs.pads_end);
}

TEST(attributes, interpolate_op4) {
    NodeBuilder::opset().insert<ov::op::v4::Interpolate>();
    auto img = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 32, 32});
    auto out_shape = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto scales = op::v0::Constant::create(element::f32, {1}, {1.0});

    op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = op::v4::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = vector<size_t>{0, 0};
    attrs.pads_end = vector<size_t>{0, 0};
    attrs.antialias = true;
    attrs.cube_coeff = -0.75;

    auto interpolate = make_shared<ov::op::v4::Interpolate>(img, out_shape, scales, attrs);
    NodeBuilder builder(interpolate, {img, out_shape, scales});
    auto g_interpolate = ov::as_type_ptr<ov::op::v4::Interpolate>(builder.create());

    const auto i_attrs = interpolate->get_attrs();
    const auto g_i_attrs = g_interpolate->get_attrs();

    EXPECT_EQ(g_i_attrs.mode, i_attrs.mode);
    EXPECT_EQ(g_i_attrs.shape_calculation_mode, i_attrs.shape_calculation_mode);
    EXPECT_EQ(g_i_attrs.coordinate_transformation_mode, i_attrs.coordinate_transformation_mode);
    EXPECT_EQ(g_i_attrs.nearest_mode, i_attrs.nearest_mode);
    EXPECT_EQ(g_i_attrs.pads_begin, i_attrs.pads_begin);
    EXPECT_EQ(g_i_attrs.pads_end, i_attrs.pads_end);
    EXPECT_EQ(g_i_attrs.antialias, i_attrs.antialias);
    EXPECT_EQ(g_i_attrs.cube_coeff, i_attrs.cube_coeff);
}

TEST(attributes, interpolate_op11) {
    NodeBuilder::opset().insert<ov::op::v11::Interpolate>();
    const auto img = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 32, 32});
    const auto scales = op::v0::Constant::create(element::f32, {4}, {1.0, 1.0, 2.0, 2.0});

    op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = op::v11::Interpolate::InterpolateMode::BILINEAR_PILLOW;
    attrs.shape_calculation_mode = op::v11::Interpolate::ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = op::v11::Interpolate::CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = vector<size_t>{0, 0};
    attrs.pads_end = vector<size_t>{0, 0};
    attrs.antialias = true;
    attrs.cube_coeff = -0.75;

    auto interpolate = make_shared<ov::op::v11::Interpolate>(img, scales, attrs);
    NodeBuilder builder(interpolate, {img, scales});
    auto g_interpolate = ov::as_type_ptr<ov::op::v11::Interpolate>(builder.create());

    const auto i_attrs = interpolate->get_attrs();
    const auto g_i_attrs = g_interpolate->get_attrs();

    EXPECT_EQ(g_i_attrs.mode, i_attrs.mode);
    EXPECT_EQ(g_i_attrs.shape_calculation_mode, i_attrs.shape_calculation_mode);
    EXPECT_EQ(g_i_attrs.coordinate_transformation_mode, i_attrs.coordinate_transformation_mode);
    EXPECT_EQ(g_i_attrs.nearest_mode, i_attrs.nearest_mode);
    EXPECT_EQ(g_i_attrs.pads_begin, i_attrs.pads_begin);
    EXPECT_EQ(g_i_attrs.pads_end, i_attrs.pads_end);
    EXPECT_EQ(g_i_attrs.antialias, i_attrs.antialias);
    EXPECT_EQ(g_i_attrs.cube_coeff, i_attrs.cube_coeff);
}
