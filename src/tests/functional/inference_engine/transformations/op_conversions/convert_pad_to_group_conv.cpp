// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <transformations/op_conversions/convert_pad_to_group_conv.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertPadToConv) {
    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        function = std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});

        manager.register_pass<pass::ConvertPadToGroupConvolution>();
    }

    {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto weights = opset4::Constant::create(element::f32, Shape{3, 1, 1, 1, 1}, {1});
        Strides stride{1, 1};
        CoordinateDiff pad_begin{1, 0}, pad_end{0, 1};
        auto conv = std::make_shared<opset4::GroupConvolution>(input, weights, stride, pad_begin, pad_end, stride);

        function_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertPadToConvNeg1) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {1, 0, 1, 0}); // Batch dim padding
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    function = get_function();
    function_ref = get_function();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
}

TEST_F(TransformationTestsF, ConvertPadToConvNeg2) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 1, 0, 1}); // Channel dim padding
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    function = get_function();
    function_ref = get_function();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
}

TEST_F(TransformationTestsF, ConvertPadToConvNeg3) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {0});
        auto pad_mode = op::PadMode::SYMMETRIC; // Unsupported mode
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    function = get_function();
    function_ref = get_function();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
}


TEST_F(TransformationTestsF, ConvertPadToConvNeg4) {
    auto get_function = []() -> std::shared_ptr<Function> {
        auto input = std::make_shared<opset4::Parameter>(element::f32, Shape{1, 3, 64, 64});
        auto pad_begin = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 1, 0});
        auto pad_end = opset4::Constant::create(element::i64, Shape{4}, {0, 0, 0, 1});
        auto pad_value = opset4::Constant::create(element::f32, Shape{}, {1.}); // Unsupported value
        auto pad_mode = op::PadMode::CONSTANT;
        auto pad = std::make_shared<opset4::Pad>(input, pad_begin, pad_end, pad_value, pad_mode);
        return std::make_shared<Function>(NodeVector{pad}, ParameterVector{input});
    };

    function = get_function();
    function_ref = get_function();
    manager.register_pass<pass::ConvertPadToGroupConvolution>();
}