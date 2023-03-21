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

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_scales) {
    {
        auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SCALES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto scales = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{2});
        const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});
        const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales, axes, attributes);
        interpolate->set_friendly_name("interpolate11");

        function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales, axes});
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
        auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SIZES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto sizes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});
        const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});
        const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, sizes, axes, attributes);
        interpolate->set_friendly_name("interpolate11");

        function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, sizes, axes});
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
        auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SCALES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto scales = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{4});
        const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales, attributes);
        interpolate->set_friendly_name("interpolate11");

        function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales});
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
        auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
        attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SIZES;
        attributes.pads_begin = {0, 0};
        attributes.pads_end = {0, 0};

        const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
        const auto sizes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{4});
        const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, sizes, attributes);
        interpolate->set_friendly_name("interpolate11");

        function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, sizes});
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

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_bicubic_pillow) {
    auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
    attributes.mode = ov::opset11::Interpolate::InterpolateMode::BICUBIC_PILLOW;
    attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SCALES;
    attributes.pads_begin = {0, 0};
    attributes.pads_end = {0, 0};

    const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
    const auto scales = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{2});
    const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});

    const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales, axes, attributes);
    interpolate->set_friendly_name("interpolate11");

    function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales, axes});
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
}

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_bilinear_pillow) {
    auto attributes = ov::opset11::Interpolate::InterpolateAttrs{};
    attributes.mode = ov::opset11::Interpolate::InterpolateMode::BILINEAR_PILLOW;
    attributes.shape_calculation_mode = ov::opset11::Interpolate::ShapeCalcMode::SCALES;
    attributes.pads_begin = {0, 0};
    attributes.pads_end = {0, 0};

    const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{1, 2, 10, 10});
    const auto scales = std::make_shared<ov::opset11::Parameter>(ov::element::f32, ov::Shape{2});
    const auto axes = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2});

    const auto interpolate = std::make_shared<ov::opset11::Interpolate>(input, scales, axes, attributes);
    interpolate->set_friendly_name("interpolate11");

    function = std::make_shared<ov::Model>(interpolate->outputs(), ov::ParameterVector{input, scales, axes});
    manager.register_pass<ov::pass::ConvertInterpolate11ToInterpolate4>();
}
