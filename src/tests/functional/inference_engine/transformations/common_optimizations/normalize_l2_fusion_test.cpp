// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/normalize_l2_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, NormalizeL2FusionWithMax) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::MAX);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {3.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithMaxIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{2}, {1, 2});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{2}, {1, 2});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAdd) {
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {eps_value});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::ADD);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectExp) {
    const float eps_value = 0.0009f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.9f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(2));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.9f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, NormalizeL2FusionWithAddIncorrectEpsValueShape) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(4));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{2}, {1, 2});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::NormalizeL2Fusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(4));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{2}, {1, 2});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }
}
