// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/opsets/opset11.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/op_conversions/convert_interpolate11_downgrade.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

namespace {
constexpr bool WITH_AXES = true;
constexpr bool WITHOUT_AXES = false;

std::shared_ptr<ov::Model> create_v11_model(const bool with_axes,
                                            const ov::opset11::Interpolate::ShapeCalcMode shape_calc_mode) {
    auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
    attributes.shape_calculation_mode = shape_calc_mode;
    attributes.pads_begin = {0, 0};
    attributes.pads_end = {0, 0};

    const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
    std::shared_ptr<ov::opset11::Parameter> scales_or_sizes;
    std::shared_ptr<ov::opset11::Interpolate> interpolate;

    const size_t num_scales_or_sizes = with_axes ? 2 : 4;
    if (shape_calc_mode == ov::opset11::Interpolate::ShapeCalcMode::SCALES) {
        scales_or_sizes = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{num_scales_or_sizes});
    } else {
        scales_or_sizes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{num_scales_or_sizes});
    }

    ov::ParameterVector model_params;
    model_params.push_back(input);
    model_params.push_back(scales_or_sizes);
    if (with_axes) {
        const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});
        model_params.push_back(axes);
        interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales_or_sizes, axes, attributes);
    } else {
        interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales_or_sizes, attributes);
    }
    interpolate->set_friendly_name("interpolate11");

    return std::make_shared<ov::Model>(interpolate->outputs(), model_params);
}
}  // namespace

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_scales) {
    {
        function = create_v11_model(WITH_AXES, ov::opset11::Interpolate::ShapeCalcMode::SCALES);
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    {
        auto attributes = ov::opset4::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset4::Interpolate::ShapeCalcMode::SCALES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto output_shape = ov::opset4::Constant::create(ov::element::i32, ov::Shape{}, {1});
        const auto scales = std::make_shared<ov::opset4::Parameter>(ov::element::f32, ov::Shape{2});
        const auto axes = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{2});

        const auto interpolate =
            std::make_shared<ov::opset4::Interpolate>(input, output_shape, scales, axes, attributes);
        interpolate->set_friendly_name("interpolate11");

        function_ref = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales, axes});
    }
}

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_sizes) {
    {
        function = create_v11_model(WITH_AXES, ov::opset11::Interpolate::ShapeCalcMode::SIZES);
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    {
        auto attributes = ov::opset4::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset4::Interpolate::ShapeCalcMode::SIZES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto output_shape = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{2});
        const auto scales = ov::opset4::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
        const auto axes = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{2});

        const auto interpolate =
            std::make_shared<ov::opset4::Interpolate>(input, output_shape, scales, axes, attributes);
        interpolate->set_friendly_name("interpolate11");

        function_ref =
            std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, output_shape, axes});
    }
}

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_no_axes) {
    {
        function = create_v11_model(WITHOUT_AXES, ov::opset11::Interpolate::ShapeCalcMode::SCALES);
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    {
        auto attributes = ov::opset4::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset4::Interpolate::ShapeCalcMode::SCALES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto output_shape = ov::opset4::Constant::create(ov::element::i32, ov::Shape{}, {1});
        const auto scales = std::make_shared<ov::opset4::Parameter>(ov::element::f32, ov::Shape{4});

        const auto interpolate = std::make_shared<ov::opset4::Interpolate>(input, output_shape, scales, attributes);
        interpolate->set_friendly_name("interpolate11");

        function_ref = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales});
    }
}

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_sizes_no_axes) {
    {
        function = create_v11_model(WITHOUT_AXES, ov::opset11::Interpolate::ShapeCalcMode::SIZES);
        manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
    }

    {
        auto attributes = ov::opset4::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset4::Interpolate::ShapeCalcMode::SIZES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto output_shape = std::make_shared<ov::opset4::Parameter>(ov::element::i32, ov::Shape{4});
        const auto scales = ov::opset4::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});

        const auto interpolate = std::make_shared<ov::opset4::Interpolate>(input, output_shape, scales, attributes);
        interpolate->set_friendly_name("interpolate11");

        function_ref = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, output_shape});
    }
}

namespace {
std::shared_ptr<ov::Model> create_non_downgradeable_model(const ov::opset11::Interpolate::InterpolateMode mode) {
    auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
    attributes.mode = mode;
    attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SCALES;
    attributes.pads_begin = {0, 0};
    attributes.pads_end = {0, 0};

    const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
    const auto scales = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{2});
    const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});

    const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales, axes, attributes);
    interpolate->set_friendly_name("interpolate11");

    return std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales, axes});
}
}  // namespace

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_bicubic_pillow) {
    function = create_non_downgradeable_model(ov::opset11::Interpolate::InterpolateMode::BICUBIC_PILLOW);
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
}

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_bilinear_pillow) {
    function = create_non_downgradeable_model(ov::opset11::Interpolate::InterpolateMode::BILINEAR_PILLOW);
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
}
