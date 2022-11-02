// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/opsets/opset9.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/control_flow/unroll_if.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

std::shared_ptr<ov::Model> get_then_body() {
    auto Xt = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto Yt = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto add_op = std::make_shared<ov::opset9::Add>(Xt, Yt);
    auto then_op_result = std::make_shared<ov::opset9::Result>(add_op);
    auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_op_result}, ov::ParameterVector{Xt, Yt});
    return then_body;
}

std::shared_ptr<ov::Model> get_else_body() {
    auto Xe = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto Ye = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto mul_op = std::make_shared<ov::opset9::Multiply>(Xe, Ye);
    auto else_op_result = std::make_shared<ov::opset9::Result>(mul_op);
    auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_op_result}, ov::ParameterVector{Xe, Ye});
    return else_body;
}

std::shared_ptr<ov::Model> create_if_model(bool condition) {
    auto X = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto Y = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
    auto cond = std::make_shared<ov::opset9::Constant>(ov::element::boolean, ov::Shape{1}, condition);
    auto if_op = std::make_shared<ov::opset8::If>(cond);
    const auto& then_body = get_then_body();
    const auto& else_body = get_else_body();

    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    auto then_p = then_body->get_parameters();
    auto else_p = else_body->get_parameters();
    if_op->set_input(X, then_p[0], else_p[0]);
    if_op->set_input(Y, then_p[1], else_p[1]);
    if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
    auto if_result = std::make_shared<ov::opset9::Result>(if_op);

    return std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
}

TEST(TransformationTests, UnrollIfCondIsTrue) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = create_if_model(true);

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_then_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfCondIsFalse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = create_if_model(false);

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_else_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfWithSplitInput) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
        auto split =
            std::make_shared<ov::opset9::Split>(X, ov::opset9::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto cond = std::make_shared<ov::opset9::Constant>(ov::element::boolean, ov::Shape{1}, false);
        auto if_op = std::make_shared<ov::opset8::If>(cond);
        const auto& then_body = get_then_body();
        const auto& else_body = get_else_body();

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();

        if_op->set_input(split->output(0), then_p[0], nullptr);
        if_op->set_input(split->output(1), nullptr, else_p[0]);
        if_op->set_input(Y, then_p[1], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<ov::opset9::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
        auto split =
            std::make_shared<ov::opset9::Split>(X, ov::opset9::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto mul_op = std::make_shared<ov::opset9::Multiply>(split->output(1), Y);
        auto if_result = std::make_shared<ov::opset9::Result>(mul_op);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollNestedIfThenBody) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<ov::opset9::Parameter>(ov::element::f32, ov::Shape{3});

        auto then_body = create_if_model(true);
        auto else_body = create_if_model(false);

        auto cond = std::make_shared<ov::opset9::Constant>(ov::element::boolean, ov::Shape{1}, true);
        auto if_op = std::make_shared<ov::opset8::If>(cond);

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(X, then_p[0], else_p[0]);
        if_op->set_input(Y, then_p[1], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<ov::opset9::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_then_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfCondIsTrueMultiOutput) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{3});
        auto X = std::make_shared<ngraph::opset6::VariadicSplit>(
            data,
            ngraph::opset6::Constant::create(ngraph::element::i32, {1}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, {2}, {1, 2}));
        auto cond = std::make_shared<ngraph::opset1::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);
        auto if_op = std::make_shared<ov::opset9::If>(cond);
        auto Xt = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto then_op_result = std::make_shared<ngraph::opset1::Result>(Xt);
        auto then_body =
            std::make_shared<ngraph::Function>(ngraph::OutputVector{then_op_result}, ngraph::ParameterVector{Xt});

        auto Xe = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{2});
        auto else_op_result = std::make_shared<ngraph::opset1::Result>(Xe);
        auto else_body =
            std::make_shared<ngraph::Function>(ngraph::OutputVector{else_op_result}, ngraph::ParameterVector{Xe});

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X->output(1), Xt, Xe);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ngraph::opset1::Result>(if_op);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{if_result}, ngraph::ParameterVector{data});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::UnrollIf>();
        manager.run_passes(f);

        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset6::Parameter>(ngraph::element::f32, ngraph::Shape{3});
        auto X = std::make_shared<ngraph::opset6::VariadicSplit>(
            data,
            ngraph::opset6::Constant::create(ngraph::element::i32, {1}, {0}),
            ngraph::opset6::Constant::create(ngraph::element::i32, {2}, {1, 2}));
        auto if_result = std::make_shared<ngraph::opset1::Result>(X->output(1));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{if_result}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
