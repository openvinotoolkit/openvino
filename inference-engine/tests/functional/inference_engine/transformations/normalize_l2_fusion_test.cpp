// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/normalize_l2_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, NormalizeL2FusionWithMax) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));

        auto multiply = std::make_shared<ngraph::opset4::Multiply>(input, input);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(multiply, axes_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto sqrt_max_eps = std::make_shared<ngraph::opset4::Maximum>(sqrt, eps_const);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt_max_eps);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NormalizeL2FusionWithMax>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::MAX);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, NormalizeL2FusionWithAdd) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    const float eps_value = 0.000099f;
    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));

        auto multiply = std::make_shared<ngraph::opset4::Multiply>(input, input);
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(multiply, axes_const);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);
        auto eps_const = ngraph::opset4::Constant::create(ngraph::element::f16, ngraph::Shape{}, {eps_value});
        auto sqrt_add_eps = std::make_shared<ngraph::opset4::Add>(sqrt, eps_const);
        auto divide = std::make_shared<ngraph::opset4::Divide>(input, sqrt_add_eps);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NormalizeL2FusionWithAdd>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f16, ngraph::PartialShape::dynamic(3));
        auto axes_const = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto normalize_l2 = std::make_shared<ngraph::opset4::NormalizeL2>(input, axes_const, eps_value, ngraph::op::EpsMode::ADD);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{normalize_l2}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
