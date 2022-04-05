// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/mvn_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, MVNFusionTestOutside) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto mean2_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean2 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean2_axes);
        auto sub2 = std::make_shared<ngraph::opset6::Subtract>(input, mean2);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub2, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(mean3, const_0_5);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(add_eps, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestReuseSub) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(mean3, const_0_5);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(add_eps, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestWithConvert) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto cast = std::make_shared<ngraph::opset6::Convert>(sub1, ngraph::element::f32);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(cast, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(mean3, const_0_5);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(add_eps, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto power_sqrt = std::make_shared<ngraph::opset6::Sqrt>(mean3);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(add_eps, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestAltDiv) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(mean3, const_0_5);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto div = std::make_shared<ngraph::opset6::Divide>(sub1, add_eps);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestInsideSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto mean2_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean2 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean2_axes);
        auto sub2 = std::make_shared<ngraph::opset6::Subtract>(input, mean2);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub2, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean3, eps);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(add_eps, const_0_5);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(power_sqrt, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestReuseSubInsideSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean3, eps);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(add_eps, const_0_5);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(power_sqrt, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestWithConvertInsideSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto cast = std::make_shared<ngraph::opset6::Convert>(sub1, ngraph::element::f32);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(cast, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean3, eps);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(add_eps, const_0_5);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(power_sqrt, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestSqrtInsideSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean3, eps);
        auto power_sqrt = std::make_shared<ngraph::opset6::Sqrt>(add_eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(power_sqrt, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestAltDivInsideSqrt) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes);
        auto sub1 = std::make_shared<ngraph::opset6::Subtract>(input, mean1);
        auto const_2 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 2 });
        auto power_sqr = std::make_shared<ngraph::opset6::Power>(sub1, const_2);
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean3, eps);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(add_eps, const_0_5);
        auto div = std::make_shared<ngraph::opset6::Divide>(sub1, power_sqrt);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }
}

TEST_F(TransformationTestsF, MVNFusionTestWithParametersInside) {
    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224 });
        auto mean1_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 1 }, { 2 });
        auto mean1 = std::make_shared<ngraph::opset6::ReduceMean>(input, mean1_axes, true);
        auto squared_difference = std::make_shared<ngraph::opset6::SquaredDifference>(input, mean1);
        auto mean2_axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 1 }, { 2 });
        auto mean2 = std::make_shared<ngraph::opset6::ReduceMean>(squared_difference, mean2_axes, true);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(mean2, eps);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(add_eps, const_0_5);
        auto gamma = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1 });
        auto mul_gamma = std::make_shared<ngraph::opset6::Multiply>(power_sqrt, gamma);
        auto mul1 = std::make_shared<ngraph::opset6::Multiply>(input, mul_gamma);
        auto mul2 = std::make_shared<ngraph::opset6::Multiply>(mul_gamma, mean1);
        auto beta = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto sub = std::make_shared<ngraph::opset6::Subtract>(beta, mul2);
        auto add = std::make_shared<ngraph::opset6::Add>(mul1, sub);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });

        manager.register_pass<ngraph::pass::MVNFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 1 }, { 2 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);
        auto gamma = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1 });
        auto mul_gamma = std::make_shared<ngraph::opset6::Multiply>(mvn, gamma);
        auto beta = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto add = std::make_shared<ngraph::opset6::Add>(mul_gamma, beta);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ add }, ngraph::ParameterVector{ input });
    }
}
