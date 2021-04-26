// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/validation_util.hpp"
#include "runtime/backend.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, if_condition_const)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
    auto Y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
    auto cond = std::make_shared<ngraph::opset5::Constant>(element::boolean, Shape{1}, true);
    auto cond2 = std::make_shared<ngraph::opset5::Constant>(element::boolean, Shape{1}, false);
    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Multiply>(Xt, Yt);
    auto result0 = make_shared<op::Result>(then_op);
    auto then_body = make_shared<ngraph::Function>(OutputVector{result0}, ParameterVector{Xt, Yt});
    auto else_body = make_shared<ngraph::Function>(OutputVector{Xe}, ParameterVector{Xe});
    auto if_op = make_shared<opset1::If>(OutputVector{cond, X, Y});
    
    opset1::If::MultiSubgraphInputDescriptionVector then_inputs = { 
        make_shared<opset1::If::InvariantInputDescription>(1, 0),
        make_shared<opset1::If::InvariantInputDescription>(2, 1),
    }; 
    opset1::If::MultiSubgraphInputDescriptionVector else_inputs = {
        make_shared<opset1::If::InvariantInputDescription>(1, 0),
    };
    opset1::If::MultiSubgraphOutputDescriptionVector outputs = {
        make_shared<opset1::If::BodyOutputDescription>(0, 0)
    };
    if_op->set_input_descriptions(if_op->then_body_index, then_inputs);
    if_op->set_input_descriptions(if_op->else_body_index, else_inputs);
    if_op->set_output_descriptions(if_op->then_body_index, outputs);
    if_op->set_output_descriptions(if_op->else_body_index, outputs);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->validate_and_infer_types();
    auto if_op2 = if_op->clone_with_new_inputs(OutputVector{cond2, X, Y});
    std::vector<float> X_v{1.0, 1.0, 1.0, 1.0};
    std::vector<float> Y_v{2.0, 2.0, 2.0, 2.0};
    auto fun = make_shared<Function>(OutputVector{if_op}, ParameterVector{X,Y});
    auto fun2 = make_shared<Function>(OutputVector{if_op2}, ParameterVector{X, Y});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                                {make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, X_v),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, Y_v)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({1, 2, 2})});
    auto result_data = read_vector<float>(result);
    std::vector<float> expected_results{2.0, 2.0, 2.0, 2.0};
    for (auto i = 0; i < expected_results.size(); i++)
        EXPECT_NEAR(result_data[i], expected_results[i], 0.000001);

    auto result1 = make_shared<HostTensor>();
    ASSERT_TRUE(fun2->evaluate({result1},
                              {make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, X_v),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, Y_v)}));
    EXPECT_EQ(result1->get_element_type(), element::f32);
    EXPECT_EQ(result1->get_shape(), Shape{std::vector<size_t>({1, 2, 2})});
    auto result_data1 = read_vector<float>(result1);
    for (auto i = 0; i < expected_results.size(); i++)
        EXPECT_NEAR(result_data1[i], X_v[i], 0.000001);
}

TEST(op_eval, if_condition_non_const)
{
    auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
    auto Y = make_shared<op::Parameter>(element::f32, Shape{1, 2, 2});
    auto cond = make_shared<op::Parameter>(element::boolean, Shape{1});
    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Multiply>(Xt, Yt);
    auto else_op = std::make_shared<op::v1::Add>(Xe, Ye);
    auto then_op_result = make_shared<op::Result>(then_op);
    auto else_op_result = make_shared<op::Result>(else_op);
    auto then_body = make_shared<ngraph::Function>(OutputVector{then_op_result}, ParameterVector{Xt, Yt});
    auto else_body =
        make_shared<ngraph::Function>(OutputVector{else_op_result}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<opset1::If>(OutputVector{cond, X, Y});

    opset1::If::MultiSubgraphInputDescriptionVector then_inputs = {
        make_shared<opset1::If::InvariantInputDescription>(1, 0),
        make_shared<opset1::If::InvariantInputDescription>(2, 1),
    };
    opset1::If::MultiSubgraphInputDescriptionVector else_inputs = {
        make_shared<opset1::If::InvariantInputDescription>(1, 0),
        make_shared<opset1::If::InvariantInputDescription>(2, 1),
    };
    opset1::If::MultiSubgraphOutputDescriptionVector outputs = {
        make_shared<opset1::If::BodyOutputDescription>(0, 0)};
    if_op->set_input_descriptions(if_op->then_body_index, then_inputs);
    if_op->set_input_descriptions(if_op->else_body_index, else_inputs);
    if_op->set_output_descriptions(if_op->then_body_index, outputs);
    if_op->set_output_descriptions(if_op->else_body_index, outputs);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->validate_and_infer_types();
    std::vector<float> X_v{1.0, 2.0, 3.0, 4.0};
    std::vector<float> Y_v{2.0, 1.0, 2.0, 3.0};
    auto fun = make_shared<Function>(OutputVector{if_op}, ParameterVector{cond, X, Y});
    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::boolean>(Shape{1}, {true}),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, X_v),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, Y_v)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({1, 2, 2})});
    auto result_data = read_vector<float>(result);
    std::vector<float> expected_results{2.0, 2.0, 6.0, 12.0};
    for (auto i = 0; i < expected_results.size(); i++)
        EXPECT_NEAR(result_data[i], expected_results[i], 0.000001);
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::boolean>(Shape{1}, {false}),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, X_v),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, Y_v)}));
    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({1, 2, 2})});
    result_data = read_vector<float>(result);
    expected_results = {3.0, 3.0, 5.0, 7.0};

    for (auto i = 0; i < expected_results.size(); i++)
        EXPECT_NEAR(result_data[i], expected_results[i], 0.000001);
}