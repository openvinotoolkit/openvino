// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/mvn6_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, MVN6Decomposition_No_Variance) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(data, axes_const, false, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MVN6Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mean = std::make_shared<ngraph::opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<ngraph::opset6::Subtract>(input0, mean);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mean_normalization }, ngraph::ParameterVector{ input0 });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, MVN6Decomposition_Inside_Sqrt) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MVN6Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mean = std::make_shared<ngraph::opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<ngraph::opset6::Subtract>(input0, mean);

        auto mul = std::make_shared<ngraph::opset6::Multiply>(mean_normalization, mean_normalization);
        auto mean2 = std::make_shared<ngraph::opset6::ReduceMean>(mul, axes_const, true);

        auto eps_node = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1e-5 });

        auto eps_add = std::make_shared<ngraph::opset6::Add>(mean2, eps_node);
        auto sqrt = std::make_shared<ngraph::opset6::Sqrt>(eps_add);
        auto div = std::make_shared<ngraph::opset6::Divide>(mean_normalization, sqrt);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input0 });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, MVN6Decomposition_Outside_Sqrt) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mvn = std::make_shared<ngraph::opset6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MVN6Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input0 = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mean = std::make_shared<ngraph::opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<ngraph::opset6::Subtract>(input0, mean);

        auto mul = std::make_shared<ngraph::opset6::Multiply>(mean_normalization, mean_normalization);
        auto mean2 = std::make_shared<ngraph::opset6::ReduceMean>(mul, axes_const, true);

        auto eps_node = ngraph::opset6::Constant::create(ngraph::element::f32, ngraph::Shape{ 1 }, { 1e-5 });

        auto sqrt = std::make_shared<ngraph::opset6::Sqrt>(mean2);
        auto eps_add = std::make_shared<ngraph::opset6::Add>(sqrt, eps_node);
        auto div = std::make_shared<ngraph::opset6::Divide>(mean_normalization, eps_add);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ div }, ngraph::ParameterVector{ input0 });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}
