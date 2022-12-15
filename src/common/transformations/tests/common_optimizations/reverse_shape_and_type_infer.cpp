// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/reverse_shape_and_type_infer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvolutionReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
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
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
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

TEST_F(TransformationTestsF, ConvolutionBackpropDataReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{20, 10, 3, 3}, std::vector<float>(20 * 10 * 3 * 3, 0.1));
        auto conv = std::make_shared<opset10::ConvolutionBackpropData>(data,
                                                                       weights,
                                                                       Strides{2, 2},
                                                                       CoordinateDiff{1, 1},
                                                                       CoordinateDiff{1, 1},
                                                                       Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto weights =
            opset10::Constant::create(element::f32, Shape{20, 10, 3, 3}, std::vector<float>(20 * 10 * 3 * 3, 0.1));
        auto conv = std::make_shared<opset10::ConvolutionBackpropData>(data,
                                                                       weights,
                                                                       Strides{2, 2},
                                                                       CoordinateDiff{1, 1},
                                                                       CoordinateDiff{1, 1},
                                                                       Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GroupConvolutionReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{3, 2, 3, 7, 7}, std::vector<float>(3 * 2 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::GroupConvolution>(data,
                                                                weights,
                                                                Strides{1, 1},
                                                                CoordinateDiff{1, 1},
                                                                CoordinateDiff{1, 1},
                                                                Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto weights =
            opset10::Constant::create(element::f32, Shape{3, 2, 3, 7, 7}, std::vector<float>(3 * 2 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::GroupConvolution>(data,
                                                                weights,
                                                                Strides{1, 1},
                                                                CoordinateDiff{1, 1},
                                                                CoordinateDiff{1, 1},
                                                                Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
}

TEST_F(TransformationTestsF, GroupConvolutionBackpropDataReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{4, 5, 2, 3, 3}, std::vector<float>(4 * 5 * 2 * 3 * 3, 0.1));
        auto conv = std::make_shared<opset10::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{2, 2},
                                                                            CoordinateDiff{1, 1},
                                                                            CoordinateDiff{1, 1},
                                                                            Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto weights =
            opset10::Constant::create(element::f32, Shape{4, 5, 2, 3, 3}, std::vector<float>(4 * 5 * 2 * 3 * 3, 0.1));
        auto conv = std::make_shared<opset10::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            Strides{2, 2},
                                                                            CoordinateDiff{1, 1},
                                                                            CoordinateDiff{1, 1},
                                                                            Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
}

TEST_F(TransformationTestsF, DeformableConvolutionReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto offsets = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 4, 5, 5}, std::vector<float>(64 * 4 * 5 * 5, 0.1));
        auto mask = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto conv = std::make_shared<opset10::DeformableConvolution>(data,
                                                                     offsets,
                                                                     weights,
                                                                     mask,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{0, 0},
                                                                     CoordinateDiff{0, 0},
                                                                     Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, offsets, mask});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto offsets = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 4, 5, 5}, std::vector<float>(64 * 4 * 5 * 5, 0.1));
        auto mask = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto conv = std::make_shared<opset10::DeformableConvolution>(data,
                                                                     offsets,
                                                                     weights,
                                                                     mask,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{0, 0},
                                                                     CoordinateDiff{0, 0},
                                                                     Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, offsets, mask});
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

TEST_F(TransformationTestsF, ActivationReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto relu = std::make_shared<opset10::Relu>(data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(relu,
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
        auto relu = std::make_shared<opset10::Relu>(data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(relu,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EltwiseScalarRightReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto add_const = opset10::Constant::create(element::f32, Shape{}, {1});
        auto add = std::make_shared<opset10::Add>(data, add_const);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(add,
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
        auto add_const = opset10::Constant::create(element::f32, Shape{}, {1});
        auto add = std::make_shared<opset10::Add>(data, add_const);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(add,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EltwiseScalarLeftReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto add_const = opset10::Constant::create(element::f32, Shape{}, {1});
        auto add = std::make_shared<opset10::Add>(add_const, data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(add,
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
        auto add_const = opset10::Constant::create(element::f32, Shape{}, {1});
        auto add = std::make_shared<opset10::Add>(add_const, data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1));
        auto conv = std::make_shared<opset10::Convolution>(add,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}
