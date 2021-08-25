// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/normalize_l2_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, NormalizeL2DecomositionFusionWithMax) {
    std::shared_ptr<ngraph::Function> f, f_ref;
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 2});
        auto normalize_l2 = std::make_shared<ngraph::opset8::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::MAX);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NormalizeL2Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset8::Power>(input, exp);
        auto axes_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 2});
        auto reduce_sum = std::make_shared<ngraph::opset8::ReduceSum>(pow, axes_const, true);
        auto eps_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset8::Maximum>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset8::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset8::Divide>(input, sqrt);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, NormalizeL2DecomositionFusionWithAdd) {
    std::shared_ptr<ngraph::Function> f, f_ref;
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<ngraph::opset8::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::ADD);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NormalizeL2Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto exp = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {2.f});
        auto pow = std::make_shared<ngraph::opset8::Power>(input, exp);
        auto axes_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset8::ReduceSum>(pow, axes_const, true);
        auto eps_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto max = std::make_shared<ngraph::opset8::Add>(reduce_sum, eps_const);
        auto sqrt = std::make_shared<ngraph::opset8::Sqrt>(max);
        auto divide = std::make_shared<ngraph::opset8::Divide>(input, sqrt);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
