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
    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Multiply>(Xt, Yt);
    auto then_body = make_shared<ngraph::Function>(OutputVector{then_op}, ParameterVector{Xt, Yt});
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
    std::vector<float> X_v{1.0, 1.0, 1.0, 1.0};
    std::vector<float> Y_v{2.0, 2.0, 2.0, 2.0};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(if_op->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, X_v),
                               make_host_tensor<element::Type_t::f32>(Shape{1, 2, 2}, Y_v)}));
/*    EXPECT_EQ(result->get_element_type(), element::f32);
    EXPECT_EQ(result->get_shape(), Shape{std::vector<size_t>({3, 2, 1})});
    auto result_data = read_vector<float>(result);
    for (auto i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);*/
}
