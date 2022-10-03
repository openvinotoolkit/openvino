// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, UnrollIfCondIsTrue) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Y = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto cond = std::make_shared<ngraph::opset1::Constant>(ngraph::element::boolean, ngraph::Shape{ 1 }, true);
        auto if_op = std::make_shared<ngraph::opset8::If>(cond);
        auto Xt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Yt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto add_op = std::make_shared<ngraph::opset1::Add>(Xt, Yt);
        auto then_op_result = std::make_shared<ngraph::opset1::Result>(add_op);
        auto then_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ then_op_result }, ngraph::ParameterVector{ Xt, Yt });

        auto Xe = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Ye = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto mul_op = std::make_shared<ngraph::opset1::Multiply>(Xe, Ye);
        auto else_op_result = std::make_shared<ngraph::opset1::Result>(mul_op);
        auto else_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ else_op_result }, ngraph::ParameterVector{ Xe, Ye });

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, Ye);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ngraph::opset1::Result>(if_op);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Y = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto add_op = std::make_shared<ngraph::opset1::Add>(X, Y);
        auto if_result = std::make_shared<ngraph::opset1::Result>(add_op);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfCondIsFalse) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Y = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto cond = std::make_shared<ngraph::opset1::Constant>(ngraph::element::boolean, ngraph::Shape{ 1 },  false);
        auto if_op = std::make_shared<ngraph::opset8::If>(cond);
        auto Xt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Yt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto add_op = std::make_shared<ngraph::opset1::Add>(Xt, Yt);
        auto then_op_result = std::make_shared<ngraph::opset1::Result>(add_op);
        auto then_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ then_op_result }, ngraph::ParameterVector{ Xt, Yt });

        auto Xe = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Ye = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto mul_op = std::make_shared<ngraph::opset1::Multiply>(Xe, Ye);
        auto else_op_result = std::make_shared<ngraph::opset1::Result>(mul_op);
        auto else_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ else_op_result }, ngraph::ParameterVector{ Xe, Ye });

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X, Xt, Xe);
        if_op->set_input(Y, Yt, Ye);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ngraph::opset1::Result>(if_op);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto Y = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto mul_op = std::make_shared<ngraph::opset1::Multiply>(X, Y);
        auto if_result = std::make_shared<ngraph::opset1::Result>(mul_op);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfWithSplitInput) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto Y = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto split = std::make_shared<ngraph::opset6::Split>(X, ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}), 2);
        auto cond = std::make_shared<ngraph::opset1::Constant>(ngraph::element::boolean, ngraph::Shape{ 1 }, false);
        auto if_op = std::make_shared<ngraph::opset8::If>(cond);
        auto Xt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3 });
        auto Yt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto add_op = std::make_shared<ngraph::opset1::Add>(Xt, Yt);
        auto then_op_result = std::make_shared<ngraph::opset1::Result>(add_op);
        auto then_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ then_op_result }, ngraph::ParameterVector{ Xt, Yt });

        auto Xe = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3 });
        auto Ye = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto mul_op = std::make_shared<ngraph::opset1::Multiply>(Xe, Ye);
        auto else_op_result = std::make_shared<ngraph::opset1::Result>(mul_op);
        auto else_body = std::make_shared<ngraph::Function>(ngraph::OutputVector{ else_op_result }, ngraph::ParameterVector{ Xe, Ye });

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(split->output(0), Xt, nullptr);
        if_op->set_input(split->output(1), nullptr, Xe);
        if_op->set_input(Y, Yt, Ye);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ngraph::opset1::Result>(if_op);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 2, 3 });
        auto Y = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto split = std::make_shared<ngraph::opset6::Split>(X, ngraph::opset6::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0}), 2);
        auto mul_op = std::make_shared<ngraph::opset1::Multiply>(split->output(1), Y);
        auto if_result = std::make_shared<ngraph::opset1::Result>(mul_op);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ if_result }, ngraph::ParameterVector{ X, Y });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
