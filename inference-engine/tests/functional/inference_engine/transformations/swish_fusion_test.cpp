// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/swish_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, SwishFusionWithBeta) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto neg = std::make_shared<ngraph::opset4::Negative>(mul);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input, beta});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input, beta});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SwishFusionWithoutBeta) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto swish = std::make_shared<ngraph::opset4::Swish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SwishFusionWithoutBetaNonOneAddConstant) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto neg = std::make_shared<ngraph::opset4::Negative>(input);
        auto exp = std::make_shared<ngraph::opset4::Exp>(neg);
        auto constant = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, constant);
        auto div = std::make_shared<ngraph::opset4::Divide>(input, add);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{div}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SwishFusionWithSigmoid) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(input);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto swish = std::make_shared<ngraph::opset4::Swish>(input);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SwishFusionWithSigmoidWithBeta) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto mul_beta = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input, beta});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::Shape{});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input, beta});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, SwishFusionWithSigmoidWithBetaConstant) {
    // test where the beta constant has multiple but the same value
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{3}, {2.0, 2.0, 2.0});
        auto mul_beta = std::make_shared<ngraph::opset4::Multiply>(input, beta);
        auto sig = std::make_shared<ngraph::opset4::Sigmoid>(mul_beta);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input, sig);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::SwishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(1));
        auto beta = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.0});
        auto swish = std::make_shared<ngraph::opset4::Swish>(input, beta);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{swish}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
