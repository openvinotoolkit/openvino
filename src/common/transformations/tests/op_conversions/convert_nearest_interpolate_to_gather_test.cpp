// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_nearest_interpolate_to_gather.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace testing;

namespace {
using CoordinateTransformMode = ov::op::util::InterpolateBase::CoordinateTransformMode;
using NearestMode = ov::op::util::InterpolateBase::NearestMode;
using ShapeCalcMode = ov::op::util::InterpolateBase::ShapeCalcMode;

std::shared_ptr<ov::Model> create_interpolate(const ov::PartialShape& shape,
                                              const std::vector<float>& scales,
                                              const std::vector<int64_t>& sizes,
                                              const std::vector<int64_t>& axes,
                                              CoordinateTransformMode ctm,
                                              NearestMode nm,
                                              ShapeCalcMode calc_mode) {
    auto attrs = ov::op::v11::Interpolate::InterpolateAttrs{};
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = calc_mode;
    attrs.coordinate_transformation_mode = ctm;
    attrs.nearest_mode = nm;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};

    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto axes_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    std::shared_ptr<ov::Node> scales_or_sizes;
    if (calc_mode == ShapeCalcMode::SCALES) {
        scales_or_sizes = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scales.size()}, scales);
    } else {
        scales_or_sizes = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{sizes.size()}, sizes);
    }
    const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, scales_or_sizes, axes_constant, attrs);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(interpolate)},
                                       ov::ParameterVector{input});
}

std::shared_ptr<ov::Model> create_gather_ref(const ov::PartialShape& shape,
                                             const std::vector<int64_t>& idx_h,
                                             const std::vector<int64_t>& idx_w) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto gather_h = std::make_shared<ov::op::v8::Gather>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{idx_h.size()}, idx_h),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2}));
    const auto gather_w = std::make_shared<ov::op::v8::Gather>(
        gather_h,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{idx_w.size()}, idx_w),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3}));
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(gather_w)},
                                       ov::ParameterVector{input});
}

const auto INPUT_2X2 = ov::PartialShape{1, 1, 2, 2};
const auto AXES_HW = std::vector<int64_t>{2, 3};
}  // namespace

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_asymmetric) {
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
    model = create_interpolate(INPUT_2X2,
                               {2.0f, 2.0f},
                               {},
                               AXES_HW,
                               CoordinateTransformMode::ASYMMETRIC,
                               NearestMode::FLOOR,
                               ShapeCalcMode::SCALES);
    model_ref = create_gather_ref(INPUT_2X2, {0, 0, 1, 1}, {0, 0, 1, 1});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_half_pixel) {
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
    model = create_interpolate(INPUT_2X2,
                               {2.0f, 2.0f},
                               {},
                               AXES_HW,
                               CoordinateTransformMode::HALF_PIXEL,
                               NearestMode::FLOOR,
                               ShapeCalcMode::SCALES);
    model_ref = create_gather_ref(INPUT_2X2, {0, 0, 0, 1}, {0, 0, 0, 1});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_sizes) {
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
    model = create_interpolate(INPUT_2X2,
                               {},
                               {4, 4},
                               AXES_HW,
                               CoordinateTransformMode::ASYMMETRIC,
                               NearestMode::FLOOR,
                               ShapeCalcMode::SIZES);
    model_ref = create_gather_ref(INPUT_2X2, {0, 0, 1, 1}, {0, 0, 1, 1});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_linear_untouched) {
    auto attrs = ov::op::v11::Interpolate::InterpolateAttrs{};
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = CoordinateTransformMode::HALF_PIXEL;
    attrs.nearest_mode = NearestMode::FLOOR;
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, INPUT_2X2);
    const auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 2.0f});
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, AXES_HW);
    const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, scales, axes, attrs);
    model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(interpolate)},
                                        ov::ParameterVector{input});
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
}

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_downscale_untouched) {
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
    model = create_interpolate(ov::PartialShape{1, 1, 4, 4},
                               {0.5f, 0.5f},
                               {},
                               AXES_HW,
                               CoordinateTransformMode::ASYMMETRIC,
                               NearestMode::FLOOR,
                               ShapeCalcMode::SCALES);
}

TEST_F(TransformationTestsF, ConvertNearestInterpolateToGather_dynamic_untouched) {
    manager.register_pass<ov::pass::ConvertNearestInterpolateToGather>();
    model = create_interpolate(ov::PartialShape{1, 1, Dimension::dynamic(), Dimension::dynamic()},
                               {2.0f, 2.0f},
                               {},
                               AXES_HW,
                               CoordinateTransformMode::ASYMMETRIC,
                               NearestMode::FLOOR,
                               ShapeCalcMode::SCALES);
}
