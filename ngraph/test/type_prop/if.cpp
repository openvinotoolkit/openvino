// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(type_prop, if_simple_test)
{
    // That which we iterate over
    auto X = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<op::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, ngraph::Shape{1}, true);

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt,Yt);
    auto then_body =
        make_shared<ngraph::Function>(OutputVector{then_op}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_body = make_shared<ngraph::Function>(OutputVector{else_op}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<op::If>(cond);
    if_op->reserve_bodies(2);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_invariant_input(Xt, X, if_op->then_body_index);
    if_op->set_invariant_input(Yt, Y, if_op->then_body_index);
    if_op->set_invariant_input(Xe, X, if_op->else_body_index);
    if_op->set_invariant_input(Ye, Y, if_op->else_body_index);
    auto res = if_op->set_output(then_op, else_op);
    auto result0 = make_shared<op::Result>(res);
    Shape out0_shape{32, 40, 10};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
    // check input descriptors
   /* for (auto& desc : tensor_iterator->get_input_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "InvariantInputDescription") == 0)
        {
            auto input_desc =
                as_type_ptr<ngraph::op::TensorIterator::InvariantInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "SliceInputDescription") == 0)
        {
            auto input_desc = as_type_ptr<ngraph::op::TensorIterator::SliceInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "MergedInputDescription") == 0)
        {
            auto input_desc = as_type_ptr<ngraph::op::TensorIterator::MergedInputDescription>(desc);
            EXPECT_NE(input_desc, nullptr);
        }
    }

    // Output 0 is last Zo
    auto out0 = tensor_iterator->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=2, part_size=2, end=38, axis=1
    auto out1 = tensor_iterator->get_concatenated_slices(Zo, 0, 2, 2, 38, 1);

    // check output descriptors
    for (auto& desc : tensor_iterator->get_output_descriptions())
    {
        auto type_info = desc->get_type_info();
        if (std::strcmp(type_info.name, "ConcatOutputDescription") == 0)
        {
            auto output_desc =
                as_type_ptr<ngraph::op::TensorIterator::ConcatOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
        else if (std::strcmp(type_info.name, "BodyOutputDescription") == 0)
        {
            auto output_desc = as_type_ptr<ngraph::op::TensorIterator::BodyOutputDescription>(desc);
            EXPECT_NE(output_desc, nullptr);
        }
    }
    auto result0 = make_shared<op::Result>(out0);
    auto result1 = make_shared<op::Result>(out1);
    Shape out0_shape{32, 2, 10};
    Shape out1_shape{32, 38, 10};

    auto results = ResultVector{result0, result1};
    auto f = make_shared<Function>(results, ParameterVector{X, Y, M});
    EXPECT_EQ(result0->get_output_shape(0), out0_shape);
    EXPECT_EQ(result1->get_output_shape(0), out1_shape);

    EXPECT_EQ(body->get_results()[0]->get_output_shape(0), out0_shape);*/
}
