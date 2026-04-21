// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/control_flow/unroll_if.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <sstream>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

using namespace testing;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;
namespace ov {
namespace test {
using op::v0::Constant;
using op::v0::Parameter;
using op::v0::Result;
using op::v1::Add;

std::shared_ptr<ov::Model> get_then_body() {
    auto Xt = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    Xt->set_friendly_name("Xt");
    auto Yt = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    Yt->set_friendly_name("Yt");
    auto add_op = std::make_shared<v1::Add>(Xt, Yt);
    add_op->set_friendly_name("add_op");
    auto then_op_result = std::make_shared<v0::Result>(add_op);
    then_op_result->set_friendly_name("then_op_result");
    auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_op_result}, ov::ParameterVector{Xt, Yt});
    ov::pass::InitNodeInfo().run_on_model(then_body);
    return then_body;
}

std::shared_ptr<ov::Model> get_else_body() {
    auto Xe = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    Xe->set_friendly_name("Xe");
    auto Ye = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    Ye->set_friendly_name("Ye");
    auto mul_op = std::make_shared<v1::Multiply>(Xe, Ye);
    mul_op->set_friendly_name("mul_op");
    auto else_op_result = std::make_shared<v0::Result>(mul_op);
    else_op_result->set_friendly_name("else_op_result");
    auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_op_result}, ov::ParameterVector{Xe, Ye});
    ov::pass::InitNodeInfo().run_on_model(else_body);
    return else_body;
}

// Generic helper functions for self-comparison tests with configurable element type and shape
std::shared_ptr<ov::Model> get_then_body_generic(ov::element::Type et, const ov::PartialShape& shape) {
    auto Xt = std::make_shared<ov::op::v0::Parameter>(et, shape);
    Xt->set_friendly_name("Xt");
    auto Yt = std::make_shared<ov::op::v0::Parameter>(et, shape);
    Yt->set_friendly_name("Yt");
    auto add_op = std::make_shared<ov::op::v1::Add>(Xt, Yt);
    add_op->set_friendly_name("add_op");
    auto then_op_result = std::make_shared<ov::op::v0::Result>(add_op);
    then_op_result->set_friendly_name("then_op_result");
    auto then_body = std::make_shared<ov::Model>(ov::OutputVector{then_op_result}, ov::ParameterVector{Xt, Yt});
    ov::pass::InitNodeInfo().run_on_model(then_body);
    return then_body;
}

std::shared_ptr<ov::Model> get_else_body_generic(ov::element::Type et, const ov::PartialShape& shape) {
    auto Xe = std::make_shared<ov::op::v0::Parameter>(et, shape);
    Xe->set_friendly_name("Xe");
    auto Ye = std::make_shared<ov::op::v0::Parameter>(et, shape);
    Ye->set_friendly_name("Ye");
    auto mul_op = std::make_shared<ov::op::v1::Multiply>(Xe, Ye);
    mul_op->set_friendly_name("mul_op");
    auto else_op_result = std::make_shared<ov::op::v0::Result>(mul_op);
    else_op_result->set_friendly_name("else_op_result");
    auto else_body = std::make_shared<ov::Model>(ov::OutputVector{else_op_result}, ov::ParameterVector{Xe, Ye});
    ov::pass::InitNodeInfo().run_on_model(else_body);
    return else_body;
}

// Backwards-compatible wrappers for i64 with Shape{3}
std::shared_ptr<ov::Model> get_then_body_i64() {
    return get_then_body_generic(ov::element::i64, ov::Shape{3});
}

std::shared_ptr<ov::Model> get_else_body_i64() {
    return get_else_body_generic(ov::element::i64, ov::Shape{3});
}

std::shared_ptr<ov::Model> create_if_model(bool condition) {
    auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    X->set_friendly_name("X");
    auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
    Y->set_friendly_name("y");
    auto cond = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, condition);
    cond->set_friendly_name("cond");
    auto if_op = std::make_shared<v8::If>(cond);
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
    auto if_result = std::make_shared<v0::Result>(if_op);
    if_result->set_friendly_name("if_result");

    return std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});
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
        if (ov::is_type<v1::Add>(op)) {
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
        if (ov::is_type<v1::Multiply>(op)) {
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
        auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto split = std::make_shared<v1::Split>(X, v0::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto cond = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, false);
        auto if_op = std::make_shared<v8::If>(cond);
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
        auto if_result = std::make_shared<v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 3});
        auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto split = std::make_shared<v1::Split>(X, v0::Constant::create(ov::element::i32, ov::Shape{}, {0}), 2);
        auto mul_op = std::make_shared<v1::Multiply>(split->output(1), Y);
        auto if_result = std::make_shared<v0::Result>(mul_op);
        f_ref = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollNestedIfThenBody) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});

        auto then_body = create_if_model(true);
        auto else_body = create_if_model(false);

        auto cond = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        auto if_op = std::make_shared<v8::If>(cond);

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(X, then_p[0], else_p[0]);
        if_op->set_input(Y, then_p[1], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});
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
        auto data = std::make_shared<v0::Parameter>(element::f32, Shape{3});
        auto X = std::make_shared<v1::VariadicSplit>(data,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {2}, {1, 2}));
        auto cond = std::make_shared<v0::Constant>(element::boolean, Shape{1}, true);
        auto if_op = std::make_shared<v8::If>(cond);
        auto Xt = std::make_shared<v0::Parameter>(element::f32, Shape{2});
        auto then_op_result = std::make_shared<v0::Result>(Xt);
        auto then_body = std::make_shared<ov::Model>(OutputVector{then_op_result}, ParameterVector{Xt});

        auto Xe = std::make_shared<v0::Parameter>(element::f32, Shape{2});
        auto else_op_result = std::make_shared<v0::Result>(Xe);
        auto else_body = std::make_shared<ov::Model>(OutputVector{else_op_result}, ParameterVector{Xe});

        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(X->output(1), Xt, Xe);
        if_op->set_output(then_op_result, else_op_result);
        auto if_result = std::make_shared<v0::Result>(if_op);

        f = std::make_shared<ov::Model>(OutputVector{if_result}, ParameterVector{data});

        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);

        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<v0::Parameter>(element::f32, Shape{3});
        auto X = std::make_shared<v1::VariadicSplit>(data,
                                                     v0::Constant::create(element::i32, {1}, {0}),
                                                     v0::Constant::create(element::i32, {2}, {1, 2}));
        auto if_result = std::make_shared<v0::Result>(X->output(1));
        f_ref = std::make_shared<ov::Model>(OutputVector{if_result}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfInsideIf) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto cond = std::make_shared<v0::Constant>(ov::element::boolean, ov::Shape{1}, true);
        auto not_cond = std::make_shared<v1::LogicalNot>(cond);
        auto if_op = std::make_shared<v8::If>(cond);

        std::shared_ptr<ov::Model> then_body;
        {
            auto cond_inside = std::make_shared<v0::Parameter>(ov::element::boolean, ov::Shape{1});
            auto if_inside = std::make_shared<v8::If>(cond_inside);

            std::shared_ptr<ov::Model> then_body_inside;
            {
                auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto add = std::make_shared<v1::Add>(X, Y);
                then_body_inside = std::make_shared<ov::Model>(add, ov::ParameterVector{X, Y});
            }
            std::shared_ptr<ov::Model> else_body_inside;
            {
                auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
                auto mul = std::make_shared<v1::Multiply>(X, Y);
                else_body_inside = std::make_shared<ov::Model>(mul, ov::ParameterVector{X, Y});
            }

            auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
            if_inside->set_then_body(then_body_inside);
            if_inside->set_else_body(else_body_inside);
            auto then_p = then_body_inside->get_parameters();
            auto else_p = else_body_inside->get_parameters();
            if_inside->set_input(X, then_p[0], else_p[0]);
            if_inside->set_input(Y, then_p[1], else_p[1]);
            if_inside->set_output(then_body_inside->get_results()[0], else_body_inside->get_results()[0]);
            auto if_result = std::make_shared<v0::Result>(if_inside);

            then_body = std::make_shared<ov::Model>(if_result, ov::ParameterVector{cond_inside, X, Y});
        }

        std::shared_ptr<ov::Model> else_body;
        {
            auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
            auto sub = std::make_shared<v1::Subtract>(X, Y);
            else_body = std::make_shared<ov::Model>(sub, ov::ParameterVector{X, Y});
        }

        auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        auto then_p = then_body->get_parameters();
        auto else_p = else_body->get_parameters();
        if_op->set_input(not_cond, then_p[0], nullptr);
        if_op->set_input(X, then_p[1], else_p[0]);
        if_op->set_input(Y, then_p[2], else_p[1]);
        if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
        auto if_result = std::make_shared<v0::Result>(if_op);

        f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::InitNodeInfo>();
        manager.register_pass<ov::pass::PushConstantToSubgraph>();
        manager.register_pass<ov::pass::ConstantFolding>();
        manager.register_pass<ov::pass::UnrollIf>();
        manager.run_passes(f);
        OV_ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto X = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto Y = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{3});
        auto mul = std::make_shared<v1::Multiply>(X, Y);
        f_ref = std::make_shared<ov::Model>(mul, ov::ParameterVector{X, Y});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, UnrollIfToParameterResultModel) {
    constexpr auto et = element::f32;
    std::shared_ptr<Model> model, model_ref;

    {
        const auto a = std::make_shared<Parameter>(et, PartialShape{5, 7});
        const auto b = std::make_shared<Parameter>(et, PartialShape{1});
        const auto c = std::make_shared<Parameter>(et, PartialShape{5, 7});

        const auto then_add = std::make_shared<Add>(a, b);
        auto then_result = std::make_shared<Result>(then_add);
        auto else_result = std::make_shared<Result>(c);

        const auto then_body = std::make_shared<Model>(OutputVector{then_result}, ParameterVector{a, b});
        const auto else_body = std::make_shared<Model>(OutputVector{else_result}, ParameterVector{c});

        const auto if_input_0 = std::make_shared<Parameter>(et, a->get_output_partial_shape(0));
        const auto if_input_1 = std::make_shared<Parameter>(et, b->get_output_partial_shape(0));
        const auto condition = Constant::create(element::boolean, {1}, {false});
        const auto if_op = std::make_shared<op::v8::If>(condition);
        if_op->set_then_body(then_body);
        if_op->set_else_body(else_body);
        if_op->set_input(if_input_0, a, c);
        if_op->set_input(if_input_1, b, nullptr);
        const auto if_result = if_op->set_output(then_result, else_result);

        const auto results = ResultVector{std::make_shared<Result>(if_result)};
        model = std::make_shared<Model>(results, ParameterVector{if_input_0, if_input_1}, "simple_if");
        model->input(0).set_names({"Input.0"});
        model->input(1).set_names({"Input.1"});
        model->output(0).set_names({"Output"});

        pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::UnrollIf>();
        manager.run_passes(model);

        OV_ASSERT_NO_THROW(check_rt_info(model));
    }
    {
        const auto p = std::make_shared<Parameter>(et, PartialShape{5, 7});
        const auto r = std::make_shared<Result>(p);
        model_ref = std::make_shared<Model>(ResultVector{r}, ParameterVector{p}, "simple_if");
        model_ref->input(0).set_names({"Input.0"});
        model_ref->output(0).set_names({"Output"});
    }

    const auto cmp_result = compare_functions(model, model_ref);
    ASSERT_TRUE(cmp_result.first) << cmp_result.second;

    EXPECT_THAT(model->input(0).get_names(), UnorderedElementsAre("Input.0", "Output"));
    EXPECT_THAT(model->output(0).get_names(), UnorderedElementsAre("Output"));
}

// Enum for comparison operations in parameterized tests
enum class ComparisonType { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };

// Parameters for self-comparison tests
struct UnrollIfSelfComparisonParams {
    ov::element::Type element_type;
    ov::PartialShape shape;
    ComparisonType comparison_type;
    bool expected_result;  // true -> then_body (Add), false -> else_body (Multiply)
};

// Helper to create comparison node based on ComparisonType
std::shared_ptr<ov::Node> create_comparison(ComparisonType type,
                                            const ov::Output<ov::Node>& lhs,
                                            const ov::Output<ov::Node>& rhs) {
    switch (type) {
    case ComparisonType::Equal:
        return std::make_shared<ov::op::v1::Equal>(lhs, rhs);
    case ComparisonType::NotEqual:
        return std::make_shared<ov::op::v1::NotEqual>(lhs, rhs);
    case ComparisonType::Less:
        return std::make_shared<ov::op::v1::Less>(lhs, rhs);
    case ComparisonType::LessEqual:
        return std::make_shared<ov::op::v1::LessEqual>(lhs, rhs);
    case ComparisonType::Greater:
        return std::make_shared<ov::op::v1::Greater>(lhs, rhs);
    case ComparisonType::GreaterEqual:
        return std::make_shared<ov::op::v1::GreaterEqual>(lhs, rhs);
    default:
        return nullptr;
    }
}

// String conversion for test naming
std::string comparison_type_to_string(ComparisonType type) {
    switch (type) {
    case ComparisonType::Equal:
        return "Equal";
    case ComparisonType::NotEqual:
        return "NotEqual";
    case ComparisonType::Less:
        return "Less";
    case ComparisonType::LessEqual:
        return "LessEqual";
    case ComparisonType::Greater:
        return "Greater";
    case ComparisonType::GreaterEqual:
        return "GreaterEqual";
    default:
        return "Unknown";
    }
}

// Parameterized test class for self-comparison tests
class UnrollIfSelfComparisonTest : public ::testing::TestWithParam<UnrollIfSelfComparisonParams> {};

TEST_P(UnrollIfSelfComparisonTest, SelfComparisonOptimization) {
    const auto& p = GetParam();

    // Create model with self-comparison condition
    auto X = std::make_shared<ov::op::v0::Parameter>(p.element_type, p.shape);
    X->set_friendly_name("X");
    auto Y = std::make_shared<ov::op::v0::Parameter>(p.element_type, p.shape);
    Y->set_friendly_name("Y");

    // Create comparison with identical inputs: X op X
    auto cond = create_comparison(p.comparison_type, X, X);
    cond->set_friendly_name("cond");

    auto if_op = std::make_shared<ov::op::v8::If>(cond);
    if_op->set_friendly_name("if_op");

    const auto& then_body = get_then_body_generic(p.element_type, p.shape);
    const auto& else_body = get_else_body_generic(p.element_type, p.shape);

    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    auto then_p = then_body->get_parameters();
    auto else_p = else_body->get_parameters();
    if_op->set_input(X, then_p[0], else_p[0]);
    if_op->set_input(Y, then_p[1], else_p[1]);
    if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);
    auto if_result = std::make_shared<ov::op::v0::Result>(if_op);
    if_result->set_friendly_name("if_result");

    auto f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.run_passes(f);

    OV_ASSERT_NO_THROW(check_rt_info(f));

    // Create reference model
    auto f_ref = p.expected_result ? get_then_body_generic(p.element_type, p.shape)
                                   : get_else_body_generic(p.element_type, p.shape);

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

// Test name generator for better readability
std::string self_comparison_test_name(const ::testing::TestParamInfo<UnrollIfSelfComparisonParams>& info) {
    const auto& p = info.param;
    std::ostringstream name;
    name << p.element_type.get_type_name() << "_";

    // Convert shape to string
    if (p.shape.is_static()) {
        name << "shape";
        for (auto dim : p.shape.get_shape()) {
            name << "_" << dim;
        }
    } else {
        name << "dynamic";
    }

    name << "_" << comparison_type_to_string(p.comparison_type);
    return name.str();
}

// Test parameters covering different element types, shapes, and comparison operations
// Self-comparison semantics (for integer types where NaN is impossible):
//   x == x  -> true    x != x  -> false
//   x <= x  -> true    x <  x  -> false
//   x >= x  -> true    x >  x  -> false
// NOTE: If condition input must have rank 0 or 1, so we test with 1D shapes only
static const std::vector<UnrollIfSelfComparisonParams> self_comparison_params = {
    // Integer types with different 1D shapes - Equal (x == x -> true)
    {ov::element::i8, ov::Shape{1}, ComparisonType::Equal, true},
    {ov::element::i16, ov::Shape{3}, ComparisonType::Equal, true},
    {ov::element::i32, ov::Shape{5}, ComparisonType::Equal, true},
    {ov::element::i64, ov::Shape{3}, ComparisonType::Equal, true},
    {ov::element::u8, ov::Shape{1}, ComparisonType::Equal, true},
    {ov::element::u16, ov::Shape{3}, ComparisonType::Equal, true},
    {ov::element::u32, ov::Shape{7}, ComparisonType::Equal, true},
    {ov::element::u64, ov::Shape{3}, ComparisonType::Equal, true},

    // NotEqual (x != x -> false)
    {ov::element::i8, ov::Shape{1}, ComparisonType::NotEqual, false},
    {ov::element::i16, ov::Shape{3}, ComparisonType::NotEqual, false},
    {ov::element::i32, ov::Shape{5}, ComparisonType::NotEqual, false},
    {ov::element::i64, ov::Shape{3}, ComparisonType::NotEqual, false},
    {ov::element::u8, ov::Shape{1}, ComparisonType::NotEqual, false},
    {ov::element::u16, ov::Shape{3}, ComparisonType::NotEqual, false},
    {ov::element::u32, ov::Shape{7}, ComparisonType::NotEqual, false},
    {ov::element::u64, ov::Shape{3}, ComparisonType::NotEqual, false},

    // Less (x < x -> false)
    {ov::element::i8, ov::Shape{1}, ComparisonType::Less, false},
    {ov::element::i32, ov::Shape{3}, ComparisonType::Less, false},
    {ov::element::i64, ov::Shape{5}, ComparisonType::Less, false},
    {ov::element::u32, ov::Shape{3}, ComparisonType::Less, false},

    // LessEqual (x <= x -> true)
    {ov::element::i8, ov::Shape{1}, ComparisonType::LessEqual, true},
    {ov::element::i32, ov::Shape{3}, ComparisonType::LessEqual, true},
    {ov::element::i64, ov::Shape{5}, ComparisonType::LessEqual, true},
    {ov::element::u32, ov::Shape{3}, ComparisonType::LessEqual, true},

    // Greater (x > x -> false)
    {ov::element::i8, ov::Shape{1}, ComparisonType::Greater, false},
    {ov::element::i32, ov::Shape{3}, ComparisonType::Greater, false},
    {ov::element::i64, ov::Shape{5}, ComparisonType::Greater, false},
    {ov::element::u32, ov::Shape{3}, ComparisonType::Greater, false},

    // GreaterEqual (x >= x -> true)
    {ov::element::i8, ov::Shape{1}, ComparisonType::GreaterEqual, true},
    {ov::element::i32, ov::Shape{3}, ComparisonType::GreaterEqual, true},
    {ov::element::i64, ov::Shape{5}, ComparisonType::GreaterEqual, true},
    {ov::element::u32, ov::Shape{3}, ComparisonType::GreaterEqual, true},
};

INSTANTIATE_TEST_SUITE_P(TransformationTests,
                         UnrollIfSelfComparisonTest,
                         ::testing::ValuesIn(self_comparison_params),
                         self_comparison_test_name);

// Test: NaN behavior - Constant with NaN values
// Per IEEE 754: NaN != NaN should be true
// When inputs are constants, the existing constant folding path (get_constant_from_source)
// correctly evaluates NaN != NaN to true per IEEE 754.
// Our self-comparison optimization is NOT triggered for constants (constant folding is used instead).
TEST(TransformationTests, UnrollIfSelfComparisonWithNaNConstant) {
    // Create a constant containing NaN
    auto nan_val = std::numeric_limits<float>::quiet_NaN();
    auto nan_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{nan_val});
    nan_const->set_friendly_name("nan_const");

    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    Y->set_friendly_name("Y");

    // NotEqual(nan_const, nan_const) - same constant source
    // Per IEEE 754: NaN != NaN -> true (should select then_body)
    // Constant folding path handles this correctly!
    auto cond = std::make_shared<ov::op::v1::NotEqual>(nan_const, nan_const);
    cond->set_friendly_name("cond");

    // Then body: output Y + 1
    std::shared_ptr<ov::Model> then_body;
    {
        auto Yt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        auto one = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{1.0f});
        auto add = std::make_shared<ov::op::v1::Add>(Yt, one);
        then_body = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{Yt});
    }

    // Else body: output Y * 2
    std::shared_ptr<ov::Model> else_body;
    {
        auto Ye = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
        auto two = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1}, std::vector<float>{2.0f});
        auto mul = std::make_shared<ov::op::v1::Multiply>(Ye, two);
        else_body = std::make_shared<ov::Model>(ov::OutputVector{mul}, ov::ParameterVector{Ye});
    }

    auto if_op = std::make_shared<ov::op::v8::If>(cond);
    if_op->set_friendly_name("if_op");
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);

    auto then_p = then_body->get_parameters();
    auto else_p = else_body->get_parameters();
    if_op->set_input(Y, then_p[0], else_p[0]);
    if_op->set_output(then_body->get_results()[0], else_body->get_results()[0]);

    auto if_result = std::make_shared<ov::op::v0::Result>(if_op);
    auto f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{Y});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.run_passes(f);

    // The If node should be removed (constant folding path evaluates condition)
    bool if_node_exists = false;
    for (const auto& op : f->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v8::If>(op)) {
            if_node_exists = true;
            break;
        }
    }
    ASSERT_FALSE(if_node_exists) << "If node should be removed";

    // Per IEEE 754, NaN != NaN is TRUE, so then_body (Add) should be selected
    bool has_add = false;
    for (const auto& op : f->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v1::Add>(op)) {
            has_add = true;
            break;
        }
    }
    ASSERT_TRUE(has_add) << "Then body (Add) should be selected since NaN != NaN is TRUE per IEEE 754";
}

// Test: Self-comparison with float Parameter is NOT optimized (IEEE 754 NaN compliance)
// Per IEEE 754: NaN != NaN is TRUE, so we cannot assume x != x is always false for floats.
// The optimization is skipped for floating-point types to preserve correct NaN semantics.
TEST(TransformationTests, UnrollIfSelfComparisonFloatParameterNotOptimized) {
    // X is a float Parameter - at runtime it COULD contain NaN
    auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    X->set_friendly_name("X");
    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Y->set_friendly_name("Y");

    // NotEqual(X, X) - same Parameter, but float type
    // IEEE 754: if X contains NaN, result should be TRUE
    // Our optimization: SKIP for float types to preserve IEEE 754 compliance
    auto cond = std::make_shared<ov::op::v1::NotEqual>(X, X);
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

    auto f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.run_passes(f);

    OV_ASSERT_NO_THROW(check_rt_info(f));

    // Verify the transformation is NOT applied for float types (If node should remain)
    bool if_node_exists = false;
    for (const auto& op : f->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v8::If>(op)) {
            if_node_exists = true;
            break;
        }
    }
    ASSERT_TRUE(if_node_exists) << "If node should NOT be removed for float self-comparison (IEEE 754 NaN safety)";
}

// Negative test: x != y should NOT be unrolled (different inputs)
TEST(TransformationTests, UnrollIfDifferentInputsNoChange) {
    auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    X->set_friendly_name("X");
    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{3});
    Y->set_friendly_name("Y");

    // Different inputs - should NOT be detected as self-comparison
    auto cond = std::make_shared<ov::op::v1::NotEqual>(X, Y);
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

    auto f = std::make_shared<ov::Model>(ov::OutputVector{if_result}, ov::ParameterVector{X, Y});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::UnrollIf>();
    manager.run_passes(f);

    // The If node should still exist since the condition is not constant
    // and not a self-comparison
    bool if_node_exists = false;
    for (const auto& op : f->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::v8::If>(op)) {
            if_node_exists = true;
            break;
        }
    }
    ASSERT_TRUE(if_node_exists) << "If node should not be removed when inputs are different";
}

}  // namespace test
}  // namespace ov
