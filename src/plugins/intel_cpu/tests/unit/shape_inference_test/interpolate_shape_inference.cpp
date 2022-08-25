// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <interpolate_shape_inference.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/interpolate.hpp>
#include <openvino/op/parameter.hpp>
#include <utils/shape_inference/shape_inference.hpp>
#include <utils/shape_inference/static_shape.hpp>

using namespace ov;
using namespace ov::intel_cpu;

using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using Nearest_mode = op::v4::Interpolate::NearestMode;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;

static std::shared_ptr<op::v4::Interpolate> build_InterpolateV4() {
    op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;

    auto input_data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());
    auto scales = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto axes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    auto interpolate = std::make_shared<op::v4::Interpolate>(input_data, sizes, scales, axes, attrs);
    return interpolate;
}

static std::shared_ptr<op::v4::Interpolate> build_InterpolateV4ConstantInput() {
    op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;

    auto input_data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto sizes = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{24, 160});
    auto scales = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{2}, std::vector<float>{2.0, 0.5});
    auto axes = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{2}, std::vector<int32_t>{2, 3});

    auto interpolate = std::make_shared<op::v4::Interpolate>(input_data, sizes, scales, axes, attrs);
    return interpolate;
}

static std::shared_ptr<op::v0::Interpolate> build_InterpolateV0() {
    ov::op::v0::Interpolate::Attributes attrs;
    attrs.axes = {2, 3};
    attrs.mode = "nearest";
    attrs.align_corners = true;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto input_data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, -1, -1, -1});
    auto sizes = std::make_shared<op::v0::Parameter>(element::i32, PartialShape::dynamic());

    auto interpolate_v0 = std::make_shared<op::v0::Interpolate>(input_data, sizes, attrs);
    return interpolate_v0;
}

TEST(StaticShapeInferenceTest, InterpolateV4Test) {
    auto interpolate = build_InterpolateV4();

    int32_t sizes_val[] = {24, 160};
    float scales_val[] = {2.0, 0.5};
    int32_t axes_val[] = {2, 3};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, sizes_val);
    constant_data[2] = std::make_shared<ngraph::runtime::HostTensor>(element::f32, Shape{2}, scales_val);
    constant_data[3] = std::make_shared<ngraph::runtime::HostTensor>(element::i32, Shape{2}, axes_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 48, 80},
                                                    StaticShape{2},
                                                    StaticShape{2},
                                                    StaticShape{2}},
                             static_output_shapes = {StaticShape{}};

    shape_inference(interpolate.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 96, 40}));
}

TEST(StaticShapeInferenceTest, InterpolateV4ConstantInputTest) {
    auto interpolate = build_InterpolateV4ConstantInput();

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 50, 80},
                                                    StaticShape{2},
                                                    StaticShape{2},
                                                    StaticShape{2}},
                             static_output_shapes = {StaticShape{}};

    shape_inference(interpolate.get(), static_input_shapes, static_output_shapes);
    ASSERT_EQ(static_output_shapes[0], StaticShape({1, 2, 100, 40}));
}

TEST(StaticShapeInferenceTest, InterpolateV4MissingConstantTest) {
    auto interpolate = build_InterpolateV4();

    int32_t sizes_val[] = {24, 160};
    float scales_val[] = {2.0, 0.5};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, sizes_val);
    constant_data[2] = std::make_shared<ngraph::runtime::HostTensor>(element::f32, Shape{2}, scales_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{1, 2, 48, 80},
                                                    StaticShape{2},
                                                    StaticShape{2},
                                                    StaticShape{2}},
                             static_output_shapes = {StaticShape{}};

    EXPECT_THROW(shape_inference(interpolate.get(), static_input_shapes, static_output_shapes, constant_data),
                 NodeValidationFailure);
}

TEST(StaticShapeInferenceTest, InterpolateV0Test) {
    auto interpolate = build_InterpolateV0();

    int32_t sizes_val[] = {15, 30};
    std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>> constant_data;
    constant_data[1] = std::make_shared<ngraph::runtime::HostTensor>(ngraph::element::Type_t::i32, Shape{2}, sizes_val);

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 2, 33, 65}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    shape_inference(interpolate.get(), static_input_shapes, static_output_shapes, constant_data);
    ASSERT_EQ(static_output_shapes[0], StaticShape({2, 2, 15, 30}));
}

TEST(StaticShapeInferenceTest, InterpolateV0MissingConstantTest) {
    auto interpolate = build_InterpolateV0();

    std::vector<StaticShape> static_input_shapes = {StaticShape{2, 2, 33, 65}, StaticShape{2}},
                             static_output_shapes = {StaticShape{}};
    EXPECT_THROW(shape_inference(interpolate.get(), static_input_shapes, static_output_shapes), NodeValidationFailure);
}