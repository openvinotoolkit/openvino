// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <queue>
#include <ngraph/op/parameter.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, EliminateSplit) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto mul_constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {89.2});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(input, mul_constant);
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {2});
        auto split = std::make_shared<ngraph::opset8::Split>(mul, axis_const, 1);
        auto res = std::make_shared<ngraph::opset8::Result>(split);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{res}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::EliminateSplit>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto mul_constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {89.2});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(input, mul_constant);
        auto res = std::make_shared<ngraph::opset8::Result>(mul);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, EliminateSplitNegative) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto mul_constant = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {89.2});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(input, mul_constant);
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {2});
        auto split = std::make_shared<ngraph::opset8::Split>(mul, axis_const, 3);
        auto res1 = std::make_shared<ngraph::opset8::Result>(split->output(0));
        auto res2 = std::make_shared<ngraph::opset8::Result>(split->output(1));
        auto res3 = std::make_shared<ngraph::opset8::Result>(split->output(2));
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{res1, res2, res3}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::EliminateSplit>();
    }
}

TEST_F(TransformationTestsF, EliminateSequenceOfSplits) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto axis_const1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto axis_const2 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        auto axis_const3 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {2});
        auto split1 = std::make_shared<ngraph::opset8::Split>(input, axis_const1, 1);
        auto split2 = std::make_shared<ngraph::opset8::Split>(split1, axis_const2, 1);
        auto split3 = std::make_shared<ngraph::opset8::Split>(split2, axis_const3, 1);
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {2});
        auto true_split = std::make_shared<ngraph::opset8::Split>(split3, axis_const, 3);
        auto res1 = std::make_shared<ngraph::opset8::Result>(true_split->output(0));
        auto res2 = std::make_shared<ngraph::opset8::Result>(true_split->output(1));
        auto res3 = std::make_shared<ngraph::opset8::Result>(true_split->output(2));
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{res1, res2, res3}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::EliminateSplit>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto axis_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {2});
        auto split = std::make_shared<ngraph::opset8::Split>(input, axis_const, 3);
        auto res1 = std::make_shared<ngraph::opset8::Result>(split->output(0));
        auto res2 = std::make_shared<ngraph::opset8::Result>(split->output(1));
        auto res3 = std::make_shared<ngraph::opset8::Result>(split->output(2));
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res1, res2, res3}, ngraph::ParameterVector{input});
    }
}