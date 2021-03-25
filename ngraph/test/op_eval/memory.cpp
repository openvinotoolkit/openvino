//*****************************************************************************
// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/util/variable.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::opset7;

TEST(op_eval, assign_readvalue_copy_test)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{3});
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(p, variable);
    auto assign = make_shared<Assign>(read_value, variable);
    auto fun = make_shared<Function>(OutputVector{assign}, ParameterVector{p});

    std::vector<int64_t> inputs{-5, 0, 5};
    std::vector<int64_t> expected_result{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::i64>(Shape{3}, inputs));
    eval_context.add_variable_value(variable, variable_value);

    auto result = make_shared<HostTensor>();
    const uint64_t COUNT_RUNS = 3;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }
}

TEST(op_eval, assign_readvalue_add)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{3});
    auto c = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int64_t>({0, 0, 0}));
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(c, variable);
    auto add = make_shared<Add>(p, read_value);
    auto assign = make_shared<Assign>(add, variable);
    auto fun = make_shared<Function>(OutputVector{assign}, ParameterVector{p});

    std::vector<int64_t> inputs{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::i64>(Shape{3}, inputs));
    eval_context.add_variable_value(variable, variable_value);

    auto result = make_shared<HostTensor>();
    const uint64_t COUNT_RUNS = 3;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        std::vector<int64_t> expected_result{inputs[0] * (i + 1), inputs[1] * (i + 1), inputs[2] * (i + 1)};
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }
}

TEST(op_eval, assign_readvalue_add_reset)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{3});
    auto c = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int64_t>({0, 0, 0}));
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(c, variable);
    auto add = make_shared<Add>(p, read_value);
    auto assign = make_shared<Assign>(add, variable);
    auto fun = make_shared<Function>(OutputVector{assign}, ParameterVector{p});

    std::vector<int64_t> inputs{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::i64>(Shape{3}, inputs));
    eval_context.add_variable_value(variable, variable_value);

    auto result = make_shared<HostTensor>();
    const uint64_t COUNT_RUNS = 3;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        std::vector<int64_t> expected_result{inputs[0] * (i + 1), inputs[1] * (i + 1), inputs[2] * (i + 1)};
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }

    eval_context.reset_variable_context();

    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        std::vector<int64_t> expected_result{inputs[0] * (i + 1), inputs[1] * (i + 1), inputs[2] * (i + 1)};
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }
}

TEST(op_eval, assign_readvalue_add_modify)
{
    auto p = make_shared<op::Parameter>(element::i64, Shape{3});
    auto c = std::make_shared<Constant>(element::i64, Shape{3}, std::vector<int64_t>({0, 0, 0}));
    auto variable = make_shared<Variable>(VariableInfo{PartialShape::dynamic(), element::dynamic, "var_1"});
    auto read_value = make_shared<ReadValue>(c, variable);
    auto add = make_shared<Add>(p, read_value);
    auto assign = make_shared<Assign>(add, variable);
    auto fun = make_shared<Function>(OutputVector{assign}, ParameterVector{p});

    std::vector<int64_t> inputs{-5, 0, 5};

    // creating context
    EvaluationContext eval_context;
    auto variable_value = make_shared<VariableValue>(make_host_tensor<element::Type_t::i64>(Shape{3}, inputs));
    eval_context.add_variable_value(variable, variable_value);

    auto result = make_shared<HostTensor>();
    const uint64_t COUNT_RUNS = 3;
    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        std::vector<int64_t> expected_result{inputs[0] * (i + 1), inputs[1] * (i + 1), inputs[2] * (i + 1)};
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }

    const auto& variables = fun->find_variables();
    EXPECT_EQ(variables.size(), 1);

    const auto& var_value = eval_context.get_variable_context().find(variables[0]);
    EXPECT_NE(var_value, eval_context.get_variable_context().end());

    var_value->second->set_value(make_host_tensor<element::Type_t::i64>(Shape{3}, {2, 2, 2}));

    for (int i = 0; i < COUNT_RUNS; ++i) {
        ASSERT_TRUE(
                fun->evaluate({result}, {make_host_tensor<element::Type_t::i64>(Shape{3}, inputs)}, eval_context));
        EXPECT_EQ(result->get_element_type(), element::i64);
        EXPECT_EQ(result->get_shape(), Shape{3});
        auto result_data = read_vector<int64_t>(result);
        std::vector<int64_t> expected_result{2 + inputs[0] * (i + 1), 2 + inputs[1] * (i + 1), 2 + inputs[2] * (i + 1)};
        for (auto j = 0; j < inputs.size(); j++)
            EXPECT_EQ(result_data[i], expected_result[i]);
    }
}
