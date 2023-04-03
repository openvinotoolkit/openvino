// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/interpolate.hpp"

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/type_prop.hpp"

using namespace ngraph;
using namespace testing;

using InterpolateMode = op::v4::Interpolate::InterpolateMode;
using CoordinateTransformMode = op::v4::Interpolate::CoordinateTransformMode;
using Nearest_mode = op::v4::Interpolate::NearestMode;
using InterpolateAttrs = op::v4::Interpolate::InterpolateAttrs;
using ShapeCalcMode = op::v4::Interpolate::ShapeCalcMode;

TEST(type_prop, interpolate_v4) {
    auto image = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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
}

TEST(type_prop, interpolate_v4_non_constant_axes_scales) {
    auto image = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<op::Parameter>(element::i64, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});

    auto axes = std::make_shared<op::Parameter>(element::i32, PartialShape{2});

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
    auto dyn_dim = Dimension::dynamic();
    auto expected_shape = PartialShape{dyn_dim, dyn_dim, dyn_dim, dyn_dim};
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(expected_shape));
}

TEST(type_prop, interpolate_v4_non_constant_axes_sizes) {
    auto image = std::make_shared<op::Parameter>(element::f32, Shape{2, 2, 30, 60});
    auto target_shape = std::make_shared<op::Parameter>(element::i64, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});

    auto axes = std::make_shared<op::Parameter>(element::i32, PartialShape{2});

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
    auto dyn_dim = Dimension::dynamic();
    auto expected_shape = PartialShape{dyn_dim, dyn_dim, dyn_dim, dyn_dim};
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(expected_shape));
}

TEST(type_prop, interpolate_v4_partial) {
    auto partial_shape = PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(partial_shape));

    // rank unknown
    auto partial_param = std::make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto interp_part = std::make_shared<op::v4::Interpolate>(partial_param, target_shape, scales, axes, attrs);
    ASSERT_TRUE(interp_part->get_output_partial_shape(0).same_scheme(PartialShape::dynamic()));
}

TEST(type_prop, interpolate_v4_partial_static_rank) {
    auto partial_shape = PartialShape{2, 2, Dimension::dynamic(), Dimension::dynamic()};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(partial_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, interpolate_v4_partial_static_rank2) {
    auto partial_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 10, 20};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 5, 10};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{15, 30});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.5f, 0.5f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(out_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, interpolate_v4_partial_static_rank3) {
    auto partial_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 3, 3};
    auto out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), 1, 1};

    auto image = std::make_shared<op::Parameter>(element::f32, partial_shape);
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{1, 1});
    auto scales = op::Constant::create<float>(element::f32, Shape{2}, {1.0f / 3.0f, 1.0f / 3.0f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(out_shape));
    ASSERT_TRUE(interp->get_output_partial_shape(0).rank().is_static());
}

TEST(type_prop, interpolate_v4_interval_logic) {
    auto image =
        std::make_shared<op::Parameter>(element::f32,
                                        PartialShape{2, 2, Dimension(12, 800), Dimension(0, -1), Dimension(24, -1)});
    auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{3});
    auto scales = op::Constant::create<float>(element::f32, Shape{3}, {0.5f, 0.25f, 0.125f});
    auto axes = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 3, 4});

    const auto out_shape = PartialShape{2, 2, Dimension(6, 400), Dimension(0, -1), Dimension(3, -1)};

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
    ASSERT_TRUE(interp->get_output_partial_shape(0).same_scheme(out_shape));
}

TEST(type_prop, interpolate_v4_incorrect_mode) {
    const auto image = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto target_shape = std::make_shared<op::Parameter>(element::i32, Shape{2});
    const auto scales = op::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

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

TEST(type_prop, interpolate_v11_scales) {
    const auto image = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = op::Constant::create<float>(element::f32, Shape{2}, {0.2f, 0.2f});
    const auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{1, 3, 6, 12}));
}

TEST(type_prop, interpolate_v11_sizes) {
    const auto image = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto sizes = op::Constant::create<int32_t>(element::i32, Shape{2}, {6, 12});
    const auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs);

    EXPECT_EQ(interp->get_element_type(), element::f32);
    EXPECT_EQ(interp->get_shape(), (Shape{1, 3, 6, 12}));
}

TEST(type_prop, interpolate_v11_scales_incorrect_et) {
    const auto image = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto scales = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 2});
    const auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};

    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, scales, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Scales element type must be f32, f16 or bf16"));
}

TEST(type_prop, interpolate_v11_sizes_incorrect_et) {
    const auto image = std::make_shared<op::Parameter>(element::f32, Shape{1, 3, 30, 60});
    const auto sizes = op::Constant::create<float>(element::f32, Shape{2}, {6.f, 12.f});
    const auto axes = op::Constant::create<int64_t>(element::i64, Shape{2}, {2, 3});

    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SIZES;
    attrs.pads_begin = {0, 0, 0, 0};
    attrs.pads_end = {0, 0, 0, 0};
    ;
    OV_EXPECT_THROW(auto interp = std::make_shared<ov::op::v11::Interpolate>(image, sizes, axes, attrs),
                    ov::NodeValidationFailure,
                    HasSubstr("Sizes element type must be i32, i64, u32 or u64"));
}
