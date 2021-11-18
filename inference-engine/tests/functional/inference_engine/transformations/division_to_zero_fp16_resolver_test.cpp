// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/division_to_zero_fp16_resolver.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
constexpr float normalized_fp16_min = 6.103515625e-05f;  // normalized minimum of fp16

TEST_F(TransformationTestsF, DivisionToZeroWithMax) {
    const float eps_value = 1.e-12;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DivisionToZeroFP16Resolver>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {normalized_fp16_min});
        auto max = std::make_shared<ngraph::opset4::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, DivisionToZeroWithAdd) {
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

        manager.register_pass<ngraph::pass::DivisionToZeroFP16Resolver>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset4::Power>(input, exp);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes_const);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {normalized_fp16_min});
        auto add = std::make_shared<ngraph::opset4::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(add);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DivisionToZeroFP16Resolver>();
    }
}
