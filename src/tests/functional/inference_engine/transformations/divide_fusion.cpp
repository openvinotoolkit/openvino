// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/divide_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, DivideFusion) {
    std::shared_ptr<ngraph::Function> f, f_ref;
    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto pow_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1});
        auto pow = std::make_shared<ngraph::opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data1, pow);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data1, data2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::DivideFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data1, data2);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data1, data2});
    }

    const auto res = FunctionsComparator::with_default()
            .enable(FunctionsComparator::CONST_VALUES)
            .enable(FunctionsComparator::ATTRIBUTES)
            .compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, DivideFusionNegative) {
    std::shared_ptr<ngraph::Function> f, f_ref;
    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto pow_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1.01});
        auto pow = std::make_shared<ngraph::opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data1, pow);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data1, data2});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::DivideFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto pow_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1.01});
        auto pow = std::make_shared<ngraph::opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data1, pow);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data1, data2});
    }

    const auto res = FunctionsComparator::with_default()
            .enable(FunctionsComparator::CONST_VALUES)
            .enable(FunctionsComparator::ATTRIBUTES)
            .compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
