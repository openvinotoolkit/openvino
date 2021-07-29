// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/opsets/opset7.hpp"
#include "ngraph/op/util/variable.hpp"
#include "ngraph/variant.hpp"
#include "ngraph/validation_util.hpp"
#include "ngraph/op/util/variable_context.hpp"

#include "util/all_close_f.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::opset7;

constexpr ngraph::VariantTypeInfo ngraph::VariantWrapper<ngraph::VariableContext>::type_info;

shared_ptr<ngraph::Function> AssignReadGraph() {
    auto p = make_shared<op::Parameter>(element::f32, Shape{3});
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(p, variable);
    auto assign = make_shared<Assign>(read_value, variable);
    return make_shared<Function>(OutputVector{assign}, ParameterVector{p}, VariableVector{variable});
}

shared_ptr<ngraph::Function> AssignReadAddGraph() {
    auto p = make_shared<op::Parameter>(element::f32, Shape{3});
    auto c = std::make_shared<Constant>(element::f32, Shape{3}, std::vector<float>({0, 0, 0}));
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(c, variable);
    auto add = make_shared<Add>(p, read_value);
    auto assign = make_shared<Assign>(add, variable);
    return make_shared<Function>(OutputVector{assign}, ParameterVector{p}, VariableVector{variable});
}

shared_ptr<ngraph::Function> AssignReadMultiVariableGraph() {
    auto c = std::make_shared<Constant>(element::f32, Shape{3}, std::vector<float>({0, 0, 0}));

    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto variable_2 = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_2"});

    auto read_value = make_shared<ReadValue>(c, variable);
    auto read_value_2 = make_shared<ReadValue>(c, variable_2);

    auto add = make_shared<Add>(read_value_2, read_value);

    auto assign = make_shared<Assign>(add, variable);
    auto assign_2 = make_shared<Assign>(read_value_2, variable_2);

    return make_shared<Function>(OutputVector{assign}, ParameterVector{}, VariableVector{variable, variable_2});
}

TEST(op_eval, assign_readvalue_without_evaluation_context)
{
    auto fun = AssignReadGraph();
    auto result = make_shared<HostTensor>();

    const int COUNT_RUNS = 10;
    std::vector<float> inputs{-5, 0, 5};
    std::vector<float> expected_result{0, 0, 0};
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_evaluation_context)
{
    auto fun = AssignReadGraph();
    auto result = make_shared<HostTensor>();
    const auto& variables = fun->get_variables();
    EXPECT_EQ(variables.size(), 1);

    std::vector<float> inputs{-5, 0, 5};
    std::vector<float> expected_result{0, 0, 0};

    EvaluationContext eval_context;
    HostTensorPtr h_tensor = make_host_tensor<element::Type_t::f32>(Shape{3}, inputs);
    VariableContext variable_context;
    variable_context.set_variable_value(variables[0], std::make_shared<VariableValue>(h_tensor));
    eval_context["VariableContext"] = std::make_shared<VariantWrapper<VariableContext>>(variable_context);

    const int COUNT_RUNS = 10;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_add)
{
    auto fun = AssignReadAddGraph();
    const auto& variables = fun->get_variables();
    EXPECT_EQ(variables.size(), 1);

    std::vector<float> inputs{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs));
    variable_context->get().set_variable_value(variables[0], variable_value);
    eval_context["VariableContext"] = variable_context;

    auto result = make_shared<HostTensor>();
    const int COUNT_RUNS = 10;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<float>(result);

        auto cnt = static_cast<float>(i+1);
        std::vector<float> expected_result{inputs[0] * cnt, inputs[1] * cnt, inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_reset_before_evaluate)
{
    auto fun = AssignReadAddGraph();
    const auto& variables = fun->get_variables();
    EXPECT_EQ(variables.size(), 1);

    std::vector<float> inputs{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs));
    variable_value->set_reset(false);
    variable_context->get().set_variable_value(variables[0], variable_value);
    eval_context["VariableContext"] = variable_context;

    auto result = make_shared<HostTensor>();
    const int COUNT_RUNS = 10;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<float>(result);

        auto cnt = static_cast<float>(i+2);
        std::vector<float> expected_result{inputs[0] * cnt, inputs[1] * cnt, inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_add_reset)
{
    auto fun = AssignReadAddGraph();
    std::vector<float> inputs{-5, 0, 5};
    const auto& variables = fun->get_variables();
    EXPECT_EQ(variables.size(), 1);

    // creating a Context
    EvaluationContext eval_context;
    auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs));
    variable_context->get().set_variable_value(variables[0], variable_value);
    eval_context["VariableContext"] = variable_context;

    auto result = make_shared<HostTensor>();
    const int COUNT_RUNS = 10;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<float>(result);

        auto cnt = static_cast<float>(i+1);
        std::vector<float> expected_result{inputs[0] * cnt, inputs[1] * cnt, inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }

    const auto& found_context = eval_context.find("VariableContext");
    EXPECT_NE(found_context, eval_context.end());
    auto var_context = std::dynamic_pointer_cast<VariantWrapper<VariableContext>>(found_context->second);
    EXPECT_NE(var_context, nullptr);
    variable_context->get().reset_variable_context();

    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<float>(result);

        auto cnt = static_cast<float>(i+1);
        std::vector<float> expected_result{inputs[0] * cnt, inputs[1] * cnt, inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_add_modify)
{
    auto fun = AssignReadAddGraph();
    std::vector<float> inputs{-5, 0, 5};
    const auto& variables = fun->get_variables();
    EXPECT_EQ(variables.size(), 1);

    // creating context
    EvaluationContext eval_context;
    auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs));
    variable_context->get().set_variable_value(variables[0], variable_value);
    eval_context["VariableContext"] = variable_context;

    auto result = make_shared<HostTensor>();
    const int COUNT_RUNS = 10;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        auto cnt = static_cast<float>(i+1);
        std::vector<float> expected_result{inputs[0] * cnt, inputs[1] * cnt, inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }

    const auto& found_context = eval_context.find("VariableContext");
    EXPECT_NE(found_context, eval_context.end());
    auto var_context = std::dynamic_pointer_cast<VariantWrapper<VariableContext>>(found_context->second);
    EXPECT_NE(var_context, nullptr);
    const auto& var_value = variable_context->get().get_variable_value(variables[0]);
    EXPECT_NE(var_value, nullptr);
    var_value->set_value(make_host_tensor<element::Type_t::f32>(Shape{3}, {1, 2, 3}));

    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::f32>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        auto cnt = static_cast<float>(i+1);
        std::vector<float> expected_result{1 + inputs[0] * cnt, 2 + inputs[1] * cnt, 3 + inputs[2] * cnt};
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}

TEST(op_eval, assign_readvalue_add_modify_multi_variables)
{
    auto fun = AssignReadMultiVariableGraph();
    std::vector<float> inputs_1{2, 2, 2};
    std::vector<float> inputs_2{1, 3, 5};
    auto var_1 = fun->get_variable_by_id("var_1");
    auto var_2 = fun->get_variable_by_id("var_2");
    EXPECT_NE(var_1, nullptr);
    EXPECT_NE(var_2, nullptr);

    // creating context
    EvaluationContext eval_context;
    auto variable_context = std::make_shared<VariantWrapper<VariableContext>>(VariableContext());
    auto variable_value_1 = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs_1));
    auto variable_value_2 = make_shared<VariableValue>(make_host_tensor<element::Type_t::f32>(Shape{3}, inputs_2));
    variable_value_1->set_reset(false);
    variable_value_2->set_reset(false);
    variable_context->get().set_variable_value(var_1, variable_value_1);
    variable_context->get().set_variable_value(var_2, variable_value_2);
    eval_context["VariableContext"] = variable_context;

    auto result = make_shared<HostTensor>();
    const int COUNT_RUNS = 10;

    std::vector<float> expected_result = inputs_1;
    for (size_t i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(fun->evaluate({result}, {}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        for (size_t j = 0; j < expected_result.size(); ++j) {
            expected_result[j] += inputs_2[j];
        }
        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }

    const auto& found_context = eval_context.find("VariableContext");
    EXPECT_NE(found_context, eval_context.end());
    auto var_context = std::dynamic_pointer_cast<VariantWrapper<VariableContext>>(found_context->second);
    EXPECT_NE(var_context, nullptr);

    auto var_value = variable_context->get().get_variable_value(var_1);
    EXPECT_NE(var_value, nullptr);
    var_value->set_value(make_host_tensor<element::Type_t::f32>(Shape{3}, {1, 2, 3}));

    auto var_value_2 = variable_context->get().get_variable_value(var_2);
    EXPECT_NE(var_value_2, nullptr);
    var_value_2->set_reset(true);

    expected_result = {1, 2, 3};
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::f32);
        EXPECT_EQ(result->get_shape(), Shape{3});

        ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
    }
}
