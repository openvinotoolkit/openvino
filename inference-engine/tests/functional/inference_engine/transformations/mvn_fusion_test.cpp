// Copyright (C) 2021 Intel Corporation
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

TEST(TransformationTests, MVNFusionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
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
        auto mean3_axes = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mean3 = std::make_shared<ngraph::opset6::ReduceMean>(power_sqr, mean3_axes);
        auto const_0_5 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 0.5 });
        auto power_sqrt = std::make_shared<ngraph::opset6::Power>(mean3, const_0_5);
        auto eps = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { 1e-9 });
        auto add_eps = std::make_shared<ngraph::opset6::Add>(power_sqrt, eps);
        auto const_neg_1 = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -1 });
        auto power_div = std::make_shared<ngraph::opset6::Power>(add_eps, const_neg_1);
        auto div = std::make_shared<ngraph::opset6::Multiply>(sub1, power_div);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MVNFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 224, 224 });
        auto axes = ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(input, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
