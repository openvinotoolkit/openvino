// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/lin_op_sequence_fusion.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, MulAddMulAddFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {3});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {4});
        auto add2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {5});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(add1, mul2_const);
        auto add2 = std::make_shared<opset3::Add>(mul2, add2_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add2}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {6});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {17});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add1}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulMulMulFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {3});
        auto mul3_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto mul2 = std::make_shared<opset3::Multiply>(mul1, mul2_const);
        auto mul3 = std::make_shared<opset3::Multiply>(mul2, mul3_const);


        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul2}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {12});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul1}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, AddAddAddFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {2});
        auto add2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {3});
        auto add3_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});

        auto add1 = std::make_shared<opset3::Add>(input, add1_const);
        auto add2 = std::make_shared<opset3::Add>(add1, add2_const);
        auto add3 = std::make_shared<opset3::Add>(add2, add3_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add3}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {8});

        auto add1 = std::make_shared<opset3::Add>(input, add1_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add1}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, MulAddAddMulFusion) {
    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {2});
        auto mul2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {3});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {4});
        auto add2_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {5});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);
        auto add2 = std::make_shared<opset3::Add>(add1, add2_const);
        auto mul2 = std::make_shared<opset3::Multiply>(add2, mul2_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul2}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::LinOpSequenceFusion>();
    }

    {
        auto input = std::make_shared<opset3::Parameter>(ngraph::element::f32, ngraph::Shape{1, 128, 3072});
        auto mul1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {10});
        auto add1_const = opset3::Constant::create(ngraph::element::f32, ngraph::Shape{128, 1}, {40});

        auto mul1 = std::make_shared<opset3::Multiply>(input, mul1_const);
        auto add1 = std::make_shared<opset3::Add>(mul1, add1_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{add1}, ngraph::ParameterVector{input});
    }
}