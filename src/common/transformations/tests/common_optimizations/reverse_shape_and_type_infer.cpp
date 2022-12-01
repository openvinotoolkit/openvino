// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/pass/manager.hpp>
#include <openvino/opsets/opset10.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/reverse_shape_and_type_infer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvolutionReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights = opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(data,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto weights = opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(data,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, PadReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto pads_begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto pads_end = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto value = opset10::Constant::create(element::f32, Shape{}, {0});
        auto pad = std::make_shared<opset10::Pad>(data, pads_begin, pads_end, value, op::PadMode::CONSTANT);
        auto result = std::make_shared<opset10::Result>(pad);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto pads_begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto pads_end = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, 0});
        auto value = opset10::Constant::create(element::f32, Shape{}, {0});
        auto pad = std::make_shared<opset10::Pad>(data, pads_begin, pads_end, value, op::PadMode::CONSTANT);
        auto result = std::make_shared<opset10::Result>(pad);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}
