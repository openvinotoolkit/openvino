// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertDivide) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertDivide>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertDivideNegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertDivide>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertDivideScalar) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        NGRAPH_CHECK(divide->get_output_partial_shape(0).rank().get_length() == 0);

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ConvertDivide>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});

        NGRAPH_CHECK(mul->get_output_partial_shape(0).rank().get_length() == 0);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
