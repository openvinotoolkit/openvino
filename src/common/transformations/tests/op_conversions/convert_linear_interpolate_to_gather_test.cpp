// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_linear_interpolate_to_gather.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"

using namespace ov;
using namespace testing;

namespace {
using CoordinateTransformMode = ov::op::util::InterpolateBase::CoordinateTransformMode;

std::shared_ptr<ov::Model> create_interpolate(const ov::PartialShape& shape,
                                              const std::vector<float>& scales,
                                              const std::vector<int64_t>& axes,
                                              CoordinateTransformMode ctm) {
    auto attrs = ov::op::v11::Interpolate::InterpolateAttrs{};
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR_ONNX;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SCALES;
    attrs.coordinate_transformation_mode = ctm;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};

    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto scales_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{scales.size()}, scales);
    const auto axes_constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, scales_constant, axes_constant, attrs);
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(interpolate)},
                                       ov::ParameterVector{input});
}

/// Builds the expected Gather/Multiply/Add chain (height axis then width axis).
std::shared_ptr<ov::Model> create_ref(const ov::PartialShape& shape,
                                      const std::vector<int64_t>& idx0,
                                      const std::vector<int64_t>& idx1,
                                      const std::vector<float>& w0,
                                      const std::vector<float>& w1) {
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    const auto out = idx0.size();
    const auto axis_h = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
    const auto axis_w = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
    const ov::Shape height_weight_shape{1, 1, out, 1};
    const ov::Shape width_weight_shape{1, 1, 1, out};

    const auto gather_h0 =
        std::make_shared<ov::op::v8::Gather>(input,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{out}, idx0),
                                             axis_h);
    const auto gather_h1 =
        std::make_shared<ov::op::v8::Gather>(input,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{out}, idx1),
                                             axis_h);
    const auto height_add = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Multiply>(gather_h0,
                                               ov::op::v0::Constant::create(ov::element::f32, height_weight_shape, w0)),
        std::make_shared<ov::op::v1::Multiply>(
            gather_h1,
            ov::op::v0::Constant::create(ov::element::f32, height_weight_shape, w1)));

    const auto gather_w0 =
        std::make_shared<ov::op::v8::Gather>(height_add,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{out}, idx0),
                                             axis_w);
    const auto gather_w1 =
        std::make_shared<ov::op::v8::Gather>(height_add,
                                             ov::op::v0::Constant::create(ov::element::i64, ov::Shape{out}, idx1),
                                             axis_w);
    const auto output = std::make_shared<ov::op::v1::Add>(
        std::make_shared<ov::op::v1::Multiply>(gather_w0,
                                               ov::op::v0::Constant::create(ov::element::f32, width_weight_shape, w0)),
        std::make_shared<ov::op::v1::Multiply>(gather_w1,
                                               ov::op::v0::Constant::create(ov::element::f32, width_weight_shape, w1)));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                       ov::ParameterVector{input});
}

const auto INPUT_2X2 = ov::PartialShape{1, 1, 2, 2};
const auto AXES_HW = std::vector<int64_t>{2, 3};
const ov::Shape OUT_SHAPE_2X2{1, 1, 4, 4};
}  // namespace

TEST_F(TransformationTestsF, ConvertLinearInterpolateToGather_half_pixel) {
    manager.register_pass<ov::pass::ConvertLinearInterpolateToGather>();
    model = create_interpolate(INPUT_2X2, {2.0f, 2.0f}, AXES_HW, CoordinateTransformMode::HALF_PIXEL);
    // coords -0.25, 0.25, 0.75, 1.25 -> clamp 0, 0.25, 0.75, 1; d2/d1 weights
    model_ref =
        create_ref(INPUT_2X2, {0, 0, 0, 1}, {1, 1, 1, 1}, {1.0f, 0.75f, 0.25f, 0.5f}, {0.0f, 0.25f, 0.75f, 0.5f});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLinearInterpolateToGather_asymmetric) {
    manager.register_pass<ov::pass::ConvertLinearInterpolateToGather>();
    model = create_interpolate(INPUT_2X2, {2.0f, 2.0f}, AXES_HW, CoordinateTransformMode::ASYMMETRIC);
    // coords 0, 0.5, 1, 1.5 -> clamp 0, 0.5, 1, 1
    model_ref = create_ref(INPUT_2X2, {0, 0, 1, 1}, {1, 1, 1, 1}, {1.0f, 0.5f, 0.5f, 0.5f}, {0.0f, 0.5f, 0.5f, 0.5f});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLinearInterpolateToGather_downscale) {
    manager.register_pass<ov::pass::ConvertLinearInterpolateToGather>();
    model =
        create_interpolate(ov::PartialShape{1, 1, 4, 4}, {0.5f, 0.5f}, AXES_HW, CoordinateTransformMode::HALF_PIXEL);
    // coords 0.5, 2.5 -> lower 0, 2 and upper 1, 3 with weights 0.5
    model_ref = create_ref(ov::PartialShape{1, 1, 4, 4}, {0, 2}, {1, 3}, {0.5f, 0.5f}, {0.5f, 0.5f});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertLinearInterpolateToGather_nearest_untouched) {
    auto attrs = ov::op::v11::Interpolate::InterpolateAttrs{};
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SCALES;
    const auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, INPUT_2X2);
    const auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 2.0f});
    const auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, AXES_HW);
    const auto interpolate = std::make_shared<ov::op::v11::Interpolate>(input, scales, axes, attrs);
    model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(interpolate)},
                                        ov::ParameterVector{input});
    manager.register_pass<ov::pass::ConvertLinearInterpolateToGather>();
}

TEST_F(TransformationTestsF, ConvertLinearInterpolateToGather_dynamic_untouched) {
    manager.register_pass<ov::pass::ConvertLinearInterpolateToGather>();
    model = create_interpolate(ov::PartialShape{1, 1, 2, ov::Dimension::dynamic()},
                               {2.0f, 2.0f},
                               AXES_HW,
                               CoordinateTransformMode::HALF_PIXEL);
}
