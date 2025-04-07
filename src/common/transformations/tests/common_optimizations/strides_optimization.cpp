// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/strides_optimization.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"

using namespace ov;
using namespace testing;

// Tests are based on model-optimizer/mo/middle/passes/fusing/resnet_optimization_test.py
// In description of unit tests below will be used next syntax: Operation(NxM,XxY), where NxM - kernel size, XxY -
// stride

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2) => Pl->Conv(1x1,2x2)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization1) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
    }
}

// Pl->Conv(3x3,2x2)->Conv(1x1,2x2) => Pl->Conv(3x3,4x4)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization2) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{4, 4},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
    }
}

// Pl->Conv(3x3,2x2)->Conv(3x3,2x2) => Same
TEST_F(TransformationTestsF, StridesOptimization3) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
    }
}

// Pl--->Conv(3x3,2x2)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(3x3,4x4)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `-->Conv(3x3,2x2)->ReLU---`                             `-->Conv(3x3,4x4)->ReLU---`
TEST_F(TransformationTestsF, StridesOptimization4) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_2 = std::make_shared<opset7::Relu>(conv_2);
        auto add = std::make_shared<opset7::Add>(relu_1, relu_2);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_3,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{4, 4},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_2,
                                                                Strides{4, 4},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_2 = std::make_shared<opset7::Relu>(conv_2);
        auto add = std::make_shared<opset7::Add>(relu_1, relu_2);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_3,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data});
    }
}

// Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(1x1,2x2)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `----------------->ReLU---`                             `-->Pool(1x1,2x2)->ReLU---`
TEST_F(TransformationTestsF, StridesOptimization5) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        auto relu_2 = std::make_shared<opset7::Relu>(data);
        auto add = std::make_shared<opset7::Add>(relu_1, relu_2);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        auto pool = std::make_shared<opset7::MaxPool>(data, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{1, 1});
        auto relu_2 = std::make_shared<opset7::Relu>(pool);
        auto add = std::make_shared<opset7::Add>(relu_1, relu_2);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_2,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_2}, ParameterVector{data});
    }
}

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2)->Conv(3x3,1x1)->Conv(1x1,2x2)
//       =>
// Pl->Conv(1x1,2x2)->Conv(1x1,1x1)->Conv(3x3,2x2)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization6) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(conv_2,
                                                                weights_3,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_4 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ov::op::v1::Convolution>(conv_3,
                                                                weights_4,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_4}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(conv_2,
                                                                weights_3,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_4 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ov::op::v1::Convolution>(conv_3,
                                                                weights_4,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_4}, ParameterVector{data});
    }
}

// Pl->Conv(1x1,1x1) --> Conv(1x1,2x2) --> Conv(1x1,2x2)
//                   `--> Relu --> Conv(1x1,2x2)
//       =>
// Pl->Conv(1x1,1x1) ---> Conv(1x1,4x4) --> Conv(1x1,1x1)
//                   `--> Pool(1x1, 2x2) -> Relu --> Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization7) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(conv_2,
                                                                weights_3,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu = std::make_shared<opset7::Relu>(conv_1);
        auto weights_4 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ov::op::v1::Convolution>(relu,
                                                                weights_4,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_3, conv_4}, ParameterVector{data});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(conv_1,
                                                                weights_2,
                                                                Strides{4, 4},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(conv_2,
                                                                weights_3,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto pool = std::make_shared<opset7::MaxPool>(conv_1, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{1, 1});
        auto relu = std::make_shared<opset7::Relu>(pool);
        auto weights_4 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ov::op::v1::Convolution>(relu,
                                                                weights_4,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_3, conv_4}, ParameterVector{data});
    }
}

// Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2)-->Eltwise-->Conv(1x1, 2x2)
//                      Const---`                    Pl---`
// =>
// Pl----->Conv(1x1,1x4)----->ReLU---->Eltwise------>Conv(1x1,1x1)------>Eltwise---->Conv(1x1, 1x1)
// Const-->MaxPool(1x1,4x4)-->Squeeze`  Pl--->MaxPool(1x1,2x2)-->Squeeze`
TEST_F(TransformationTestsF, StridesOptimization8) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        Shape const_shape{1, 3, 224, 224};
        auto constant =
            opset7::Constant::create(element::f32, const_shape, std::vector<float>(shape_size(const_shape), 1));
        auto add = std::make_shared<opset7::Add>(relu_1, constant);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 112, 112});
        auto add_2 = std::make_shared<opset7::Add>(conv_2, data_2);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add_2,
                                                                weights_3,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data, data_2});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{4, 4},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto relu_1 = std::make_shared<opset7::Relu>(conv_1);
        Shape const_shape{1, 3, 56, 56};
        auto constant =
            opset7::Constant::create(element::f32, const_shape, std::vector<float>(shape_size(const_shape), 1));
        auto add = std::make_shared<opset7::Add>(relu_1, constant);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add,
                                                                weights_2,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 112, 112});
        auto reshape =
            std::make_shared<opset7::Reshape>(data_2,
                                              opset7::Constant::create(element::i64, Shape{4}, {1, 3, 112, 112}),
                                              false);
        auto pool_2 = std::make_shared<opset7::MaxPool>(reshape, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{1, 1});
        auto squeeze = std::make_shared<opset7::Squeeze>(pool_2, op::v0::Constant::create(element::u64, Shape{1}, {0}));
        auto add_2 = std::make_shared<opset7::Add>(conv_2, squeeze);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add_2,
                                                                weights_3,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data, data_2});
    }
}

// Pl------->Conv(1x1,1x1)------>Eltwise------>Conv(1x1,2x2)---->Eltwise-->Conv(1x1, 2x2)
// Pl----->Eltwise---->Eltwise--`  Pl--->Eltwise------>Eltwise--`
// Const--`      Const-`          Const--`     Const-`
// =>
// Pl------->Conv(1x1,4x4)------->Eltwise---->Conv(1x1,1x1)-->Eltwise-->Conv(1x1, 1x1)
// Pl----->Eltwise----->Eltwise--`    Eltwise------>Eltwise--`
// Const--`      Const-`        Const--`     Const-`
TEST_F(TransformationTestsF, StridesOptimization9) {
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{});

        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{224});
        auto add_const = ov::op::v0::Constant::create(element::f32, Shape{224}, {128});
        auto add = std::make_shared<opset7::Add>(data_2, add_const);
        auto add_2_const = ov::op::v0::Constant::create(element::f32, Shape{224}, {128});
        auto add_2 = std::make_shared<opset7::Add>(add, add_2_const);

        auto add_3 = std::make_shared<opset7::Add>(conv_1, add_2);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add_3,
                                                                weights_2,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        auto data_3 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto add_4_const = ov::op::v0::Constant::create(element::f32, Shape{}, {128});
        auto add_4 = std::make_shared<opset7::Add>(data_3, add_4_const);
        auto add_5_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {128});
        auto add_5 = std::make_shared<opset7::Add>(add_4, add_5_const);
        auto add_6 = std::make_shared<opset7::Add>(conv_2, add_5);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add_6,
                                                                weights_3,
                                                                Strides{2, 2},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data, data_2, data_3});
        manager.register_pass<ov::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 3, 224, 224});
        auto weights_1 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ov::op::v1::Convolution>(data,
                                                                weights_1,
                                                                Strides{4, 4},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{});

        auto data_2 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{224});
        auto reshape =
            std::make_shared<opset7::Reshape>(data_2,
                                              opset7::Constant::create(element::i64, Shape{4}, {1, 1, 1, 224}),
                                              false);
        auto pool = std::make_shared<opset7::MaxPool>(reshape, Strides{4, 4}, Shape{0, 0}, Shape{0, 0}, Shape{1, 1});
        auto squeeze =
            std::make_shared<opset7::Squeeze>(pool, op::v0::Constant::create(element::u64, Shape{3}, {0, 1, 2}));
        auto add_const = ov::op::v0::Constant::create(element::f32, Shape{56}, {128});
        auto add = std::make_shared<opset7::Add>(squeeze, add_const);
        auto add_2_const = ov::op::v0::Constant::create(element::f32, Shape{56}, {128});
        auto add_2 = std::make_shared<opset7::Add>(add, add_2_const);

        auto add_3 = std::make_shared<opset7::Add>(conv_1, add_2);
        auto weights_2 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ov::op::v1::Convolution>(add_3,
                                                                weights_2,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        auto data_3 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
        auto new_shape = ov::op::v0::Constant::create(element::i64, Shape{4}, {1, 1, 1, 1});
        auto reshape_2 = std::make_shared<opset7::Reshape>(data_3, new_shape, false);
        auto pool_2 =
            std::make_shared<opset7::MaxPool>(reshape_2, Strides{2, 2}, Shape{0, 0}, Shape{0, 0}, Shape{1, 1});
        auto squeeze_2 =
            std::make_shared<opset7::Squeeze>(pool_2, op::v0::Constant::create(element::u64, Shape{4}, {0, 1, 2, 3}));
        auto add_4_const = ov::op::v0::Constant::create(element::f32, Shape{}, {128});
        auto add_4 = std::make_shared<opset7::Add>(squeeze_2, add_4_const);
        auto add_5_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {128});
        auto add_5 = std::make_shared<opset7::Add>(add_4, add_5_const);
        auto add_6 = std::make_shared<opset7::Add>(conv_2, add_5);
        auto weights_3 = ov::op::v0::Constant::create(element::f32, Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ov::op::v1::Convolution>(add_6,
                                                                weights_3,
                                                                Strides{1, 1},
                                                                CoordinateDiff{},
                                                                CoordinateDiff{},
                                                                Strides{});

        model_ref = std::make_shared<ov::Model>(NodeVector{conv_3}, ParameterVector{data, data_2, data_3});
    }
}
