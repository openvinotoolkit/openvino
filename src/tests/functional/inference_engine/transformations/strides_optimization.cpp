// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/common_optimizations/strides_optimization.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// Tests are based on model-optimizer/mo/middle/passes/fusing/resnet_optimization_test.py
// In description of unit tests below will be used next syntax: Operation(NxM,XxY), where NxM - kernel size, XxY - stride

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2) => Pl->Conv(1x1,2x2)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization1) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }
}

// Pl->Conv(3x3,2x2)->Conv(1x1,2x2) => Pl->Conv(3x3,4x4)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization2) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }
}

// Pl->Conv(3x3,2x2)->Conv(3x3,2x2) => Same
TEST_F(TransformationTestsF, StridesOptimization3) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }
}

// Pl--->Conv(3x3,2x2)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(3x3,4x4)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `-->Conv(3x3,2x2)->ReLU---`                             `-->Conv(3x3,4x4)->ReLU---`
TEST_F(TransformationTestsF, StridesOptimization4) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(data, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(conv_2);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(data, weights_2, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(conv_2);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data});
    }
}

// Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2) => Pl--->Conv(1x1,2x2)->ReLU--->Eltwise-->Conv(1x1,1x1)
//   `----------------->ReLU---`                             `-->Pool(1x1,2x2)->ReLU---`
TEST_F(TransformationTestsF, StridesOptimization5) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(data);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(data, ngraph::Strides{2, 2}, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto relu_2 = std::make_shared<ngraph::opset7::Relu>(pool);
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, relu_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_2}, ngraph::ParameterVector{data});
    }

    // TODO: update transformation and remove this check XXX-68696
    disable_rt_info_check();
}

// Pl->Conv(1x1,1x1)->Conv(1x1,2x2)->Conv(3x3,1x1)->Conv(1x1,2x2)
//       =>
// Pl->Conv(1x1,2x2)->Conv(1x1,1x1)->Conv(3x3,2x2)->Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization6) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(conv_3, weights_4, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_4}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 3, 3}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(conv_3, weights_4, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_4}, ngraph::ParameterVector{data});
    }
}

// Pl->Conv(1x1,1x1) --> Conv(1x1,2x2) --> Conv(1x1,2x2)
//                   `--> Relu --> Conv(1x1,2x2)
//       =>
// Pl->Conv(1x1,1x1) ---> Conv(1x1,4x4) --> Conv(1x1,1x1)
//                   `--> Pool(1x1, 2x2) -> Relu --> Conv(1x1,1x1)
TEST_F(TransformationTestsF, StridesOptimization7) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu = std::make_shared<ngraph::opset7::Relu>(conv_1);
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(relu, weights_4, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3, conv_4}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(conv_1, weights_2, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(conv_2, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(conv_1, ngraph::Strides{2, 2}, ngraph::Shape{0, 0}, ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto relu = std::make_shared<ngraph::opset7::Relu>(pool);
        auto weights_4 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_4 = std::make_shared<ngraph::opset7::Convolution>(relu, weights_4, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3, conv_4}, ngraph::ParameterVector{data});
    }
    // TODO: update transformation and remove this check XXX-68696
    disable_rt_info_check();
}

// Pl--->Conv(1x1,1x1)->ReLU--->Eltwise-->Conv(1x1,2x2)-->Eltwise-->Conv(1x1, 2x2)
//                      Const---`                    Pl---`
// =>
// Pl----->Conv(1x1,1x4)----->ReLU---->Eltwise------>Conv(1x1,1x1)------>Eltwise---->Conv(1x1, 1x1)
// Const-->MaxPool(1x1,4x4)-->Squeeze`  Pl--->MaxPool(1x1,2x2)-->Squeeze`
TEST_F(TransformationTestsF, StridesOptimization8) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        ngraph::Shape const_shape{1, 3, 224, 224};
        auto constant = ngraph::opset7::Constant::create(ngraph::element::f32, const_shape, std::vector<float>(shape_size(const_shape), 1));
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, constant);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto data_2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 112, 112});
        auto add_2 = std::make_shared<ngraph::opset7::Add>(conv_2, data_2);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add_2, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data, data_2});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto relu_1 = std::make_shared<ngraph::opset7::Relu>(conv_1);
        ngraph::Shape const_shape{1, 3, 56, 56};
        auto constant = ngraph::opset7::Constant::create(ngraph::element::f32, const_shape, std::vector<float>(shape_size(const_shape), 1));
        auto add = std::make_shared<ngraph::opset7::Add>(relu_1, constant);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});
        auto data_2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 112, 112});
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(data_2,
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 3, 112, 112}), false);
        auto pool_2 = std::make_shared<ngraph::opset7::MaxPool>(reshape, ngraph::Strides{2, 2}, ngraph::Shape{0, 0},
                ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto squeeze = std::make_shared<ngraph::opset7::Squeeze>(pool_2,
                ngraph::op::Constant::create(ngraph::element::u64, ngraph::Shape{1}, {0}));
        auto add_2 = std::make_shared<ngraph::opset7::Add>(conv_2, squeeze);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add_2, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data, data_2});
    }
    // TODO: update transformation and remove this check XXX-68696
    disable_rt_info_check();
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
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{});

        auto data_2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{224});
        auto add_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{224}, {128});
        auto add = std::make_shared<ngraph::opset7::Add>(data_2, add_const);
        auto add_2_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{224}, {128});
        auto add_2 = std::make_shared<ngraph::opset7::Add>(add, add_2_const);

        auto add_3 = std::make_shared<ngraph::opset7::Add>(conv_1, add_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add_3, weights_2, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        auto data_3 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto add_4_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {128});
        auto add_4 = std::make_shared<ngraph::opset7::Add>(data_3, add_4_const);
        auto add_5_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {128});
        auto add_5 = std::make_shared<ngraph::opset7::Add>(add_4, add_5_const);
        auto add_6 = std::make_shared<ngraph::opset7::Add>(conv_2, add_5);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add_6, weights_3, ngraph::Strides{2, 2},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data, data_2, data_3});
        manager.register_pass<ngraph::pass::StridesOptimization>();
    }
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 224, 224});
        auto weights_1 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_1 = std::make_shared<ngraph::opset7::Convolution>(data, weights_1, ngraph::Strides{4, 4},
                ngraph::CoordinateDiff{0, 0}, ngraph::CoordinateDiff{0, 0}, ngraph::Strides{});

        auto data_2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{224});
        auto reshape = std::make_shared<ngraph::opset7::Reshape>(data_2,
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 224}), false);
        auto pool = std::make_shared<ngraph::opset7::MaxPool>(reshape, ngraph::Strides{4, 4}, ngraph::Shape{0, 0},
                ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto squeeze = std::make_shared<ngraph::opset7::Squeeze>(pool,
                ngraph::op::Constant::create(ngraph::element::u64, ngraph::Shape{3}, {0, 1, 2}));
        auto add_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{56}, {128});
        auto add = std::make_shared<ngraph::opset7::Add>(squeeze, add_const);
        auto add_2_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{56}, {128});
        auto add_2 = std::make_shared<ngraph::opset7::Add>(add, add_2_const);

        auto add_3 = std::make_shared<ngraph::opset7::Add>(conv_1, add_2);
        auto weights_2 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_2 = std::make_shared<ngraph::opset7::Convolution>(add_3, weights_2, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        auto data_3 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto new_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {1, 1, 1, 1});
        auto reshape_2 = std::make_shared<ngraph::opset7::Reshape>(data_3, new_shape, false);
        auto pool_2 = std::make_shared<ngraph::opset7::MaxPool>(reshape_2, ngraph::Strides{2, 2}, ngraph::Shape{0, 0},
                ngraph::Shape{0, 0}, ngraph::Shape{1, 1});
        auto squeeze_2 = std::make_shared<ngraph::opset7::Squeeze>(pool_2,
                ngraph::op::Constant::create(ngraph::element::u64, ngraph::Shape{4}, {0, 1, 2, 3}));
        auto add_4_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {128});
        auto add_4 = std::make_shared<ngraph::opset7::Add>(squeeze_2, add_4_const);
        auto add_5_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {128});
        auto add_5 = std::make_shared<ngraph::opset7::Add>(add_4, add_5_const);
        auto add_6 = std::make_shared<ngraph::opset7::Add>(conv_2, add_5);
        auto weights_3 = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3, 3, 1, 1}, {128});
        auto conv_3 = std::make_shared<ngraph::opset7::Convolution>(add_6, weights_3, ngraph::Strides{1, 1},
                ngraph::CoordinateDiff{}, ngraph::CoordinateDiff{}, ngraph::Strides{});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv_3}, ngraph::ParameterVector{data, data_2, data_3});
    }
    // TODO: update transformation and remove this check XXX-68696
    disable_rt_info_check();
}
