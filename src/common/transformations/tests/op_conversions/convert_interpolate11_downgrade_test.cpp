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

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4) {
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

TEST_F(TransformationTestsF, ConvertInterpolate11ToInterpolate4_fail) {
    // const auto input = std::make_shared<ov::opset11::Parameter>(ov::element::i32, ov::Shape{2, 3, 4});
    // const auto k = std::make_shared<ov::opset11::Parameter>(ov::element::i8, ov::Shape{});
    // const auto topk = std::make_shared<ov::opset11::TopK>(input,
    //                                                       k,
    //                                                       -2,
    //                                                       ov::op::TopKMode::MAX,
    //                                                       ov::op::TopKSortType::SORT_VALUES,
    //                                                       ov::element::i64,
    //                                                       true);  // stable sort on
    // topk->set_friendly_name("topk11");

    // function = std::make_shared<ov::Model>(topk->outputs(), ov::ParameterVector{input, k});
    // manager.register_pass<ov::pass::ConvertTopK11ToTopK3>();
}
