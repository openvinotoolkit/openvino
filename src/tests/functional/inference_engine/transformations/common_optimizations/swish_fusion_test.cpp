// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/swish_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, SwishFusionWithBeta) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto neg = std::make_shared<ngraph::opset4::Negative>(mul);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input, beta});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input, beta});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithoutBeta) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto swish = std::make_shared<ngraph::opset4::Swish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithoutBetaNonOneAddConstant) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoid) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(input);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto swish = std::make_shared<ngraph::opset4::Swish>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoidWithBeta) {
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto mul_beta = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input, beta});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input, beta});
    }
}

TEST_F(TransformationTestsF, SwishFusionWithSigmoidWithBetaConstant) {
    // test where the beta constant has multiple but the same value
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{3}, {2.0, 2.0, 2.0});
        auto mul_beta = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::SwishFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.0});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }
}
