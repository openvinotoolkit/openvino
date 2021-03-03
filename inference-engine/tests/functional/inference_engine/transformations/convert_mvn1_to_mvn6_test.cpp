// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_mvn1_to_mvn6.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertMVN1ToMVN6) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 2, 3 });
        auto mvn = std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMVN1ToMVN6_across_channels) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, true, true, 1e-5);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 1, 2, 3 });
        auto mvn = std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertMVN1ToMVN6_5D) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset2::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4, 5 });
        auto mvn = std::make_shared<ngraph::op::v0::MVN>(data, false, true, 1e-5);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertMVN1ToMVN6>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 2, 3, 4, 5 });
        auto axes_const = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 2, 3, 4 });
        auto mvn = std::make_shared<ngraph::op::v6::MVN>(data, axes_const, true, 1e-5, ngraph::op::MVNEpsMode::INSIDE_SQRT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ mvn }, ngraph::ParameterVector{ data });
    }

    auto res = compare_functions(f, f_ref, false, false, false, false);
    ASSERT_TRUE(res.first) << res.second;
}
