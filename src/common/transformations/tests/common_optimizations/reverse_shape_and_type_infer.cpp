// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reverse_shape_and_type_infer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset10.hpp"
#include "openvino/opsets/opset12.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvolutionReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, DYN, DYN});
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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

TEST_F(TransformationTestsF, ConvolutionReverseInferUpdateShape) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic(4));
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, DYN, DYN});
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{20, 10, 3, 3}, std::vector<float>(20 * 10 * 3 * 3, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 20, DYN, DYN});
        auto weights =
            opset10::Constant::create(element::f32, Shape{20, 10, 3, 3}, std::vector<float>(20 * 10 * 3 * 3, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{3, 2, 3, 7, 7}, std::vector<float>(3 * 2 * 3 * 7 * 7, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 9, DYN, DYN});
        auto weights =
            opset10::Constant::create(element::f32, Shape{3, 2, 3, 7, 7}, std::vector<float>(3 * 2 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{4, 5, 2, 3, 3}, std::vector<float>(4 * 5 * 2 * 3 * 3, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 20, DYN, DYN});
        auto weights =
            opset10::Constant::create(element::f32, Shape{4, 5, 2, 3, 3}, std::vector<float>(4 * 5 * 2 * 3 * 3, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 4, 5, 5}, std::vector<float>(64 * 4 * 5 * 5, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 4, 5, 5}, std::vector<float>(64 * 4 * 5 * 5, 0.1f));
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

TEST_F(TransformationTestsF, NegativePad12ReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto pads_begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pads_end = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto value = opset10::Constant::create(element::f32, Shape{}, {0});
        auto pad = std::make_shared<ov::op::v12::Pad>(data, pads_begin, pads_end, value, op::PadMode::CONSTANT);
        auto result = std::make_shared<opset10::Result>(pad);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto pads_begin = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto pads_end = opset10::Constant::create(element::i64, Shape{4}, {0, 0, 0, -1});
        auto value = opset10::Constant::create(element::f32, Shape{}, {0});
        auto pad = std::make_shared<ov::op::v12::Pad>(data, pads_begin, pads_end, value, op::PadMode::CONSTANT);
        auto result = std::make_shared<opset10::Result>(pad);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ActivationReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto relu = std::make_shared<opset10::Relu>(data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, DYN, DYN});
        auto relu = std::make_shared<opset10::Relu>(data);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
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

TEST_F(TransformationTestsF, ConcatReverseInfer) {
    {
        auto data1 = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        // Specify rank and type in one of Concat input to inherit in another
        auto data2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 224, 224});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{data1, data2}, 0);
        auto result = std::make_shared<opset10::Result>(concat);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data1 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, 224, 224});
        auto data2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, 3, 224, 224});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{data1, data2}, 0);
        auto result = std::make_shared<opset10::Result>(concat);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
    }
}

TEST_F(TransformationTestsF, ConcatReverseInferUpdateShape) {
    {
        auto data1 = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape{DYN, DYN, 224, DYN});
        // Specify rank and type in one of Concat input to inherit in another
        auto data2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, DYN, 224});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{data1, data2}, 0);
        auto result = std::make_shared<opset10::Result>(concat);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data1 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, 224, 224});
        auto data2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, 3, 224, 224});
        auto concat = std::make_shared<opset10::Concat>(OutputVector{data1, data2}, 0);
        auto result = std::make_shared<opset10::Result>(concat);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
    }
}

TEST_F(TransformationTestsF, SliceReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto start = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto stop = opset10::Constant::create(element::i32, Shape{1}, {1});
        auto step = opset10::Constant::create(element::i32, Shape{1}, {1});
        auto axis = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto slice = std::make_shared<opset10::Slice>(data, start, stop, step, axis);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(slice,
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
        auto start = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto stop = opset10::Constant::create(element::i32, Shape{1}, {1});
        auto step = opset10::Constant::create(element::i32, Shape{1}, {1});
        auto axis = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto slice = std::make_shared<opset10::Slice>(data, start, stop, step, axis);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(slice,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SqueezeReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto axes = opset10::Constant::create(element::i32, Shape{2}, {0, 1});
        auto squeeze = std::make_shared<opset10::Squeeze>(data, axes);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(squeeze,
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(6));
        auto axes = opset10::Constant::create(element::i32, Shape{2}, {0, 1});
        auto squeeze = std::make_shared<opset10::Squeeze>(data, axes);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(squeeze,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, SqueezeAxesReverseInfer) {
    auto dyn = Dimension::dynamic();
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape{1, dyn, 1, dyn, dyn, dyn});
        auto squeeze = std::make_shared<opset10::Squeeze>(data);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(squeeze,
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{1, dyn, 1, dyn, dyn, dyn});
        auto axes = opset10::Constant::create(element::i64, Shape{2}, {0, 2});
        auto squeeze = std::make_shared<opset10::Squeeze>(data, axes);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(squeeze,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, UnsqueezeReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto axes = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto unsqueeze = std::make_shared<opset10::Unsqueeze>(data, axes);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(unsqueeze,
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(3));
        auto axes = opset10::Constant::create(element::i32, Shape{1}, {0});
        auto unsqueeze = std::make_shared<opset10::Unsqueeze>(data, axes);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(unsqueeze,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertLikeReverseInfer) {
    {
        // One input has static rank and another has static type
        auto data1 = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto data2 = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto convert_like = std::make_shared<opset10::ConvertLike>(data1, data2);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(convert_like,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data1 = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape{DYN, 3, DYN, DYN});
        auto data2 = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic());
        auto convert_like = std::make_shared<opset10::ConvertLike>(data1, data2);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(convert_like,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data1, data2});
    }
}

TEST_F(TransformationTestsF, IfReverseInfer) {
    {
        auto X = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto cond = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());

        // Body parameters
        auto Xt = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto Xe = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        // Body
        auto one = opset10::Constant::create(element::f32, Shape{}, {1});
        auto then_op = std::make_shared<opset10::Add>(Xt, one);
        auto then_op_res = std::make_shared<opset10::Result>(then_op);
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt});

        auto neg_one = opset10::Constant::create(element::f32, Shape{}, {-1});
        auto else_op = std::make_shared<opset10::Add>(Xe, neg_one);
        auto else_op_res = std::make_shared<opset10::Result>(else_op);
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe});

        auto if_op = std::make_shared<opset10::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        auto res = if_op->set_output(then_op_res, else_op_res);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(res,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{X, cond});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto X = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto cond = std::make_shared<opset10::Parameter>(element::boolean, PartialShape::dynamic());

        // Body parameters
        auto Xt = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto Xe = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        // Body
        auto one = opset10::Constant::create(element::f32, Shape{}, {1});
        auto then_op = std::make_shared<opset10::Add>(Xt, one);
        auto then_op_res = std::make_shared<opset10::Result>(then_op);
        auto then_body = std::make_shared<Model>(OutputVector{then_op_res}, ParameterVector{Xt});

        auto neg_one = opset10::Constant::create(element::f32, Shape{}, {-1});
        auto else_op = std::make_shared<opset10::Add>(Xe, neg_one);
        auto else_op_res = std::make_shared<opset10::Result>(else_op);
        auto else_body = std::make_shared<Model>(OutputVector{else_op_res}, ParameterVector{Xe});

        auto if_op = std::make_shared<opset10::If>(cond);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        auto res = if_op->set_output(then_op_res, else_op_res);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(res,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{X, cond});
    }
}

TEST_F(TransformationTestsF, TransposeWithDynamicOrderReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto order = std::make_shared<opset10::Parameter>(element::i32, PartialShape::dynamic());
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(transpose,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, order});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape::dynamic(4));
        auto order = std::make_shared<opset10::Parameter>(element::i32, PartialShape::dynamic());
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        // Convolution is needed to produce static rank
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(transpose,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data, order});
    }
}

TEST_F(TransformationTestsF, TransposeWithConstantOrderReverseInfer) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto order = std::make_shared<opset10::Constant>(element::i32, Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        auto result = std::make_shared<opset10::Result>(transpose);
        model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
        manager.register_pass<pass::ReverseShapeAndTypeInfer>();
    }
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic(3));
        auto order = std::make_shared<opset10::Constant>(element::i32, Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        auto result = std::make_shared<opset10::Result>(transpose);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, TransposeWithConstantOrderReverseInfer2) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::dynamic, PartialShape::dynamic());
        auto order = std::make_shared<opset10::Constant>(element::i32, Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(transpose,
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
        auto data = std::make_shared<opset10::Parameter>(element::f32, PartialShape{DYN, DYN, DYN, 3});
        auto order = std::make_shared<opset10::Constant>(element::i32, Shape{4}, std::vector<int64_t>{0, 3, 1, 2});
        auto transpose = std::make_shared<opset10::Transpose>(data, order);
        auto weights =
            opset10::Constant::create(element::f32, Shape{64, 3, 7, 7}, std::vector<float>(64 * 3 * 7 * 7, 0.1f));
        auto conv = std::make_shared<opset10::Convolution>(transpose,
                                                           weights,
                                                           Strides{2, 2},
                                                           CoordinateDiff{3, 3},
                                                           CoordinateDiff{3, 3},
                                                           Strides{1, 1});
        auto result = std::make_shared<opset10::Result>(conv);
        model_ref = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    }
}
