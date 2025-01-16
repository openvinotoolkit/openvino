// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/control_flow/unroll_if.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using namespace ov;
using namespace testing;

std::shared_ptr<ov::Model> get_then_body() {
    auto Xt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Xt->set_friendly_name("Xt");
    auto Yt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Yt->set_friendly_name("Yt");
    auto add_op = std::make_shared<ov::op::v1::Add>(Xt, Yt);
    add_op->set_friendly_name("add_op");
    auto then_op_result = std::make_shared<ov::op::v0::Result>(add_op);
    then_op_result->set_friendly_name("then_op_result");
    auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_op_result}, ov::ParameterVector{Xt, Yt});
    ov::pass::InitNodeInfo().run_on_model(then_body);
    return then_body;
}

std::shared_ptr<ov::Model> get_else_body() {
    auto Xe = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Xe->set_friendly_name("Xe");
    auto Ye = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Ye->set_friendly_name("Ye");
    auto mul_op = std::make_shared<ov::op::v1::Multiply>(Xe, Ye);
    mul_op->set_friendly_name("mul_op");
    auto else_op_result = std::make_shared<ov::op::v0::Result>(mul_op);
    else_op_result->set_friendly_name("else_op_result");
    auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_op_result}, ov::ParameterVector{Xe, Ye});
    ov::pass::InitNodeInfo().run_on_model(else_body);
    return else_body;
}

std::shared_ptr<ov::Model> create_if_model(bool condition) {
    auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    X->set_friendly_name("X");
    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Y->set_friendly_name("y");
    auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, condition);
    cond->set_friendly_name("cond");
    auto if_op = std::make_shared<ov::op::v8::If>(cond);
    if_op->set_friendly_name("if_op");
    const auto& then_body = get_then_body();
    const auto& else_body = get_else_body();

    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    auto then_p = then_body->get_parameters();
    auto else_p = else_body->get_parameters();
    if_op->set_input(X, then_p[0], else_p[0]);
    if_op->set_input(Y, then_p[1], else_p[1]);
    if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
    auto if_result = std::make_shared<ov::op::v0::Result>(if_op);
    if_result->set_friendly_name("if_result");

    return std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
}

TEST(TransformationTests, UnrollIfCondIsTrue) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = create_if_model(true);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_then_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    for (auto& op : f->get_ops()) {
        std::vector<std::string> fused_names = ov::getFusedNamesVector(op);
        if (ov::is_type<ov::op::v1::Add>(op)) {
            ASSERT_EQ(2, fused_names.size());
            ASSERT_TRUE(ov::util::contains(fused_names, "add_op"));
            ASSERT_TRUE(ov::util::contains(fused_names, "if_op"));
        } else {
            ASSERT_EQ(1, fused_names.size());
            ASSERT_TRUE(!ov::util::contains(fused_names, "if_op"));
        }
    }
}

TEST(TransformationTests, UnrollIfCondIsFalse) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        f = create_if_model(false);

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_else_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    for (auto& op : f->get_ops()) {
        std::vector<std::string> fused_names = ov::getFusedNamesVector(op);
        if (ov::is_type<ov::op::v1::Multiply>(op)) {
            ASSERT_EQ(2, fused_names.size());
            ASSERT_TRUE(ov::util::contains(fused_names, "mul_op"));
            ASSERT_TRUE(ov::util::contains(fused_names, "if_op"));
        } else {
            ASSERT_EQ(1, fused_names.size());
            ASSERT_TRUE(!ov::util::contains(fused_names, "if_op"));
        }
    }
}

TEST(TransformationTests, UnrollIfWithSplitInput) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto split =
            std::make_shared<ov::op::v1::Split>(X, ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, false);
        auto if_op = std::make_shared<ov::op::v8::If>(cond);
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
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto split =
            std::make_shared<ov::op::v1::Split>(X, ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto mul_op = std::make_shared<ov::op::v1::Multiply>(split->output(1), Y);
        auto if_result = std::make_shared<ov::op::v0::Result>(mul_op);
        f_ref = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollNestedIfThenBody) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});

        auto then_body = create_if_model(true);
        auto else_body = create_if_model(false);

        auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        auto if_op = std::make_shared<ov::op::v8::If>(cond);

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(X, then_p[0], else_p[0]);
        if_op->set_input(Y, then_p[1], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    { f_ref = get_then_body(); }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfCondIsTrueMultiOutput) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
        auto X = std::make_shared<ov::op::v1::VariadicSplit>(data,
                                                             ov::op::v0::Constant::create(element::i32, {1}, {0}),
                                                             ov::op::v0::Constant::create(element::i32, {2}, {1, 2}));
        auto cond = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
        auto if_op = std::make_shared<ov::op::v8::If>(cond);
        auto Xt = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
        auto then_op_result = std::make_shared<ov::op::v0::Result>(Xt);
        auto then_body = std::make_shared<ov::Model>(OutputVector{then_op_result}, ParameterVector{Xt});

        auto Xe = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
        auto else_op_result = std::make_shared<ov::op::v0::Result>(Xe);
        auto else_body = std::make_shared<ov::Model>(OutputVector{else_op_result}, ParameterVector{Xe});

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X->output(1), Xt, Xe);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        f = std::make_shared<ov::Model>(NodeVector{if_result}, ParameterVector{data});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{3});
        auto X = std::make_shared<ov::op::v1::VariadicSplit>(data,
                                                             ov::op::v0::Constant::create(element::i32, {1}, {0}),
                                                             ov::op::v0::Constant::create(element::i32, {2}, {1, 2}));
        auto if_result = std::make_shared<ov::op::v0::Result>(X->output(1));
        f_ref = std::make_shared<ov::Model>(NodeVector{if_result}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfInsideIf) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        auto not_cond = std::make_shared<ov::op::v1::LogicalNot>(cond);
        auto if_op = std::make_shared<ov::op::v8::If>(cond);

        std::shared_ptr<ov::Model> then_body;
        {
            auto cond_inside = std::make_shared<ov::op::v0::Parameter>(ov::element::boolean, ov::Shape{1});
            auto if_inside = std::make_shared<ov::op::v8::If>(cond_inside);

            std::shared_ptr<ov::Model> then_body_inside;
            {
                auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto add = std::make_shared<ov::op::v1::Add>(X, Y);
                then_body_inside = std::make_shared<ov::Model>(add, ov::ParameterVector{X, Y});
            }
            std::shared_ptr<ov::Model> else_body_inside;
            {
                auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto mul = std::make_shared<ov::op::v1::Multiply>(X, Y);
                else_body_inside = std::make_shared<ov::Model>(mul, ov::ParameterVector{X, Y});
            }

            auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
            if_inside->set_then_body(then_body_inside);
            if_inside->set_else_body(else_body_inside);
            auto then_p = then_body_inside->get_parameters();
            auto else_p = else_body_inside->get_parameters();
            if_inside->set_input(X, then_p[0], else_p[0]);
            if_inside->set_input(Y, then_p[1], else_p[1]);
            if_inside->set_output(then_body_inside->get_results()[0], else_body_inside->get_results()[0]);
            auto if_result = std::make_shared<ov::op::v0::Result>(if_inside);

            then_body = std::make_shared<ov::Model>(if_result, ov::ParameterVector{cond_inside, X, Y});
        }

        std::shared_ptr<ov::Model> else_body;
        {
            auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto sub = std::make_shared<ov::op::v1::Subtract>(X, Y);
            else_body = std::make_shared<ov::Model>(sub, ov::ParameterVector{X, Y});
        }

        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(not_cond, then_p[0], nullptr);
        if_op->set_input(X, then_p[1], else_p[0]);
        if_op->set_input(Y, then_p[2], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<ov::op::v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::NodeVector{if_result}, ov::ParameterVector{X, Y});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::PushConstantToSubgraph>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto mul = std::make_shared<ov::op::v1::Multiply>(X, Y);
        f_ref = std::make_shared<ov::Model>(mul, ov::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
