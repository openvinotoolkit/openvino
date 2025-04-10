// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"

using namespace ov;
using namespace testing;

using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using Nearest_mode = op::v4::Interpolate::NearestMode;
using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;

TEST(type_prop, interpolate_v0_default_ctor) {
    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = ov::op::v0::Constant::create<float>(element::i32, Shape{2}, {15, 30});

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = AxisSet{2, 3};
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto interp = std::make_shared<op::v0::Interpolate>();
    interp->set_arguments(OutputVector{image, target_shape});
    interp->set_attrs(attrs);
    interp->validate_and_infer_types();

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{2, 2, 15, 30}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v0_all_inputs_dynamic_rank) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape::dynamic());
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = AxisSet{2, 3};
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto interp = std::make_shared<ov::op::v0::Interpolate>(image, target_shape, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, interpolate_v0_all_inputs_static_rank) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape::dynamic(6));
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic(1));

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = AxisSet{2, 3};
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto interp = std::make_shared<ov::op::v0::Interpolate>(image, target_shape, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic(6));
}

TEST(type_prop, interpolate_v0_target_shape_not_constant) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::bf16, PartialShape{2, 4, 12, 12});
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape{1});

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = AxisSet{3, 1};
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    auto interp = std::make_shared<ov::op::v0::Interpolate>(image, target_shape, attrs);

    EXPECT_EQ(interp->get_element_type(), element::bf16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({2, -1, 12, -1}));
}

TEST(type_prop, interpolate_v0_target_shape_as_shape_of) {
    auto img_shape = PartialShape{{1, 2}, 10, 10, {5, 30}};
    auto out_shape = PartialShape{{2, 4}, -1};
    auto img_symbols = set_shape_symbols(img_shape);
    auto out_symbols = set_shape_symbols(out_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f64, img_shape);
    auto target_shape =
        std::make_shared<op::v0::ShapeOf>(std::make_shared<ov::op::v0::Parameter>(element::i32, out_shape));

    op::v0::Interpolate::Attributes attrs;
    attrs.axes = AxisSet{3, 1};
    attrs.pads_begin = {0, 0, 1, 0};
    attrs.pads_end = {0, 2, 0, 0};
    auto interp = std::make_shared<op::v0::Interpolate>(image, target_shape, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f64);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({{1, 2}, {2, 4}, 10, -1}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(img_symbols[0], out_symbols[0], img_symbols[2], out_symbols[1]));
}

// --- v4 ---
TEST(type_prop, interpolate_v4_default_ctor) {
    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    attrs.cube_coeff = -0.75;

    auto interp = std::make_shared<op::v4::Interpolate>();
    interp->set_arguments(OutputVector{image, target_shape, scales, axes});
    interp->set_attrs(attrs);
    interp->validate_and_infer_types();

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{2, 2, 15, 30}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v4) {
    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{15, 30});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{2, 2, 15, 30}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v4_non_constant_axes_scales) {
    auto img_shape = PartialShape{2, 2, 30, 60};
    set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f16, img_shape);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{2});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v4_non_constant_axes_sizes) {
    auto img_shape = PartialShape{2, 2, 30, 60};
    set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::bf16, img_shape);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i64, Shape{2});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2, 1}, {0.5f, 0.5f});

    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{2});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::bf16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v4_img_dynamic_rank) {
    auto image = std::make_shared<ov::op::v0::Parameter>(element::bf16, PartialShape::dynamic());
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::bf16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, interpolate_v4_partial_static_rank) {
    auto img_shape = PartialShape{2, 2, -1, {5, 30}};
    auto symbols = set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 1, 0, 0};
    attrs.pads_end = {0, 1, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({2, 4, -1, {2, 15}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], nullptr, nullptr, nullptr));
}

TEST(type_prop, interpolate_v4_img_intervals_use_scales) {
    auto img_shape = PartialShape{{1, 2}, -1, 10, {5, 30}};
    set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 1};
    attrs.pads_end = {1, 1, 0, 1};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({{2, 3}, {1, -1}, 5, {3, 16}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v4_use_sizes_as_shape_of) {
    auto img_shape = PartialShape{{1, 2}, 10, 10, {5, 30}};
    auto out_shape = PartialShape{{2, 4}, -1};
    auto img_symbols = set_shape_symbols(img_shape);
    auto out_symbols = set_shape_symbols(out_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    auto target_shape =
        std::make_shared<op::v0::ShapeOf>(std::make_shared<ov::op::v0::Parameter>(element::i32, out_shape));
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {1.0f / 3.0f, 1.0f / 3.0f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {3, 1});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({{1, 2}, -1, 10, {2, 4}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(img_symbols[0], out_symbols[1], img_symbols[2], out_symbols[0]));
}

TEST(type_prop, interpolate_v4_use_scales_interval_shapes) {
    auto img_shape = PartialShape{2, 2, {12, 800}, {0, -1}, {24, -1}};
    auto symbols = set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{3});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{3}, {0.5f, 0.25f, 0.125f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 4});

    InterpolateAttrs attrs;
    attrs.mode = InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = Nearest_mode::ROUND_PREFER_FLOOR;
    attrs.antialias = false;
    attrs.pads_begin = {0, 0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0, 0};
    attrs.cube_coeff = -0.75;
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({2, 2, {6, 400}, -1, {3, -1}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr, nullptr));
}

TEST(type_prop, interpolate_v4_target_shapes_gt_axes_number) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = ov::op::v0::Constant::create<float>(element::i32, Shape{3}, {10, 12, 20});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{1}, {0.3f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {0, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({10, 3, 30, 12}));
}

TEST(type_prop, interpolate_v4_scales_gt_axes_number) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{3});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{3}, {0.2f, 0.2f, 0.3f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<op::v4::Interpolate>(image, target_shape, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({1, 3, 6, 12}));
}

TEST(type_prop, interpolate_v4_incorrect_mode) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{2});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::BICUBIC_PILLOW;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v4::Interpolate>(image, target_shape, scales, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Unsupported interpolation mode used with version 4 of the Interpolate op"));

    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::BILINEAR_PILLOW;
    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v4::Interpolate>(image, target_shape, scales, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Unsupported interpolation mode used with version 4 of the Interpolate op"));
}

TEST(type_prop, interpolate_v4_target_shape_not_1d) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::NEAREST;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2}),
                        scales,
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [1] is not rank 1"));

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{}),
                        scales,
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [1] is not rank 1"));
}

TEST(type_prop, interpolate_v4_scales_not_1d) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = ov::op::v0::Constant::create<float>(element::i32, Shape{2}, {10, 20});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::NEAREST;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        target_shape,
                        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2}),
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [2] is not rank 1"));

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        target_shape,
                        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{}),
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [2] is not rank 1"));
}

TEST(type_prop, interpolate_v4_axes_not_1d) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = ov::op::v0::Constant::create<float>(element::i32, Shape{2}, {10, 20});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::NEAREST;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        target_shape,
                        scales,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2}),
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [3] is not rank 1"));

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v4::Interpolate>(
                        image,
                        target_shape,
                        scales,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2}),
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [3] is not rank 1"));
}

// --- v11 ---
TEST(type_prop, interpolate_v11_default_ctor) {
    auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.2f, 0.2f});
    auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = std::vector<size_t>{1, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 1, 0, 0};

    auto interp = std::make_shared<op::v11::Interpolate>();
    interp->set_arguments(OutputVector{image, scales, axes});
    interp->set_attrs(attrs);
    interp->validate_and_infer_types();

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{2, 4, 6, 12}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v11_scales) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.2f, 0.2f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{1, 3, 6, 12}));
}

TEST(type_prop, interpolate_v11_scales_all_inputs_static_rank) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f16, PartialShape::dynamic(8));
    const auto scales = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    const auto axes = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic(1));

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic(8));
}

TEST(type_prop, interpolate_v11_sizes) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto sizes = ov::op::v0::Constant::create<int32_t>(element::i32, Shape{2}, {6, 12});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{1, 3, 6, 12}));
}

TEST(type_prop, interpolate_v11_sizes_all_inputs_dynamic_rank) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto sizes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape::dynamic());
    const auto axes = std::make_shared<ov::op::v0::Parameter>(element::i64, PartialShape::dynamic());

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST(type_prop, interpolate_v11_intervals_with_scales_mode) {
    auto img_shape = PartialShape{{1, 3}, 3, {1, 10}, {10, -1}, {10, 20}};
    auto symbols = set_shape_symbols(img_shape);

    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{3}, {2.0f, 3.0f, 1.0f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 4});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 2, 1, 0, 0};
    attrs.pads_end = {1, 1, 0, 1, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({{2, 4}, 6, {4, 22}, {33, -1}, {10, 20}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(nullptr, nullptr, nullptr, nullptr, symbols[4]));
}

TEST(type_prop, interpolate_v11_intervals_with_sizes_mode) {
    auto img_shape = PartialShape{{1, 3}, 3, {1, 10}, {10, -1}};
    auto symbols = set_shape_symbols(img_shape);

    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    const auto sizes = ov::op::v0::Constant::create<float>(element::i32, Shape{2}, {200, 300});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs);

    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape({{1, 3}, 3, 200, 300}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(symbols[0], symbols[1], nullptr, nullptr));
}

TEST(type_prop, interpolate_v11_sizes_with_shapeof) {
    auto img_shape = PartialShape{{1, 3}, 3, {1, 10}, {10, -1}};
    auto sizes_shape = PartialShape{{12, 37}, {0, 21}};
    auto img_symbols = set_shape_symbols(img_shape);
    auto sizes_symbols = set_shape_symbols(sizes_shape);

    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, img_shape);
    const auto param = std::make_shared<ov::op::v0::Parameter>(element::f32, sizes_shape);
    const auto sizes = std::make_shared<op::v0::ShapeOf>(param);
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 1});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs);

    EXPECT_EQ(interp->get_output_partial_shape(0), (PartialShape{{1, 3}, {0, 21}, {12, 37}, {10, -1}}));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)),
                ElementsAre(img_symbols[0], sizes_symbols[1], sizes_symbols[0], img_symbols[3]));
}

TEST(type_prop, interpolate_v11_non_constant_axes_scales) {
    auto img_shape = PartialShape{2, 2, 30, 60};
    set_shape_symbols(img_shape);

    auto image = std::make_shared<ov::op::v0::Parameter>(element::f16, img_shape);
    auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = std::make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{2});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    auto interp = std::make_shared<op::v11::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f16);
    EXPECT_EQ(interp->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_THAT(get_shape_symbols(interp->get_output_partial_shape(0)), Each(nullptr));
}

TEST(type_prop, interpolate_v11_scales_incorrect_et) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 2});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};

    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Scales element type must be f32, f16 or bf16"));
}

TEST(type_prop, interpolate_v11_sizes_incorrect_et) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto sizes = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};

    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Sizes element type must be i32, i64, u32 or u64"));
}

TEST(type_prop, interpolate_v11_scales_incorrect_number) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {0.2f, 0.2f});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("The number of elements in the 'scales' input does not match the number of axes"));
}

TEST(type_prop, interpolate_v11_sizes_incorrect_number) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto sizes = ov::op::v0::Constant::create<int32_t>(element::i32, Shape{2}, {6, 12});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};
    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("The number of elements in the 'sizes' input does not match the number of axes"));
}

TEST(type_prop, interpolate_v11_scales_not_1d) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto axes = ov::op::v0::Constant::create<int64_t>(element::i32, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v11::Interpolate>(
                        image,
                        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2}),
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [1] is not rank 1"));

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v11::Interpolate>(
                        image,
                        std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{}),
                        axes,
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [1] is not rank 1"));
}

TEST(type_prop, interpolate_v11_axes_not_1d) {
    const auto image = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = ov::op::v0::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::NEAREST;
    attrs.pads_begin = std::vector<size_t>{0, 0, 0, 0};
    attrs.pads_end = std::vector<size_t>{0, 0, 0, 0};

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v11::Interpolate>(
                        image,
                        scales,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 2}),
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [2] is not rank 1"));

    OV_EXPECT_THROW(std::ignore = std::make_shared<ov::op::v11::Interpolate>(
                        image,
                        scales,
                        std::make_shared<ov::op::v0::Parameter>(element::i32, Shape{}),
                        attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Input [2] is not rank 1"));
}
