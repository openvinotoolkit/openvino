// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/result.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, if_simple_test) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, true);

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto convert_then_op = std::make_shared<op::v0::Convert>(then_op, element::f32);
    auto then_op_res = std::make_shared<op::v0::Result>(convert_then_op);

    auto then_body = make_shared<ov::Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto convert_else_op = std::make_shared<op::v0::Convert>(else_op, element::f32);
    auto else_op_res = std::make_shared<ov::op::v0::Result>(convert_else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_op_res, else_op_res);
    auto result0 = make_shared<op::v0::Result>(res);
    Shape out0_shape{32, 40, 10};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
    // Check that If validation when condition is constant validates single body
    convert_then_op->set_convert_element_type(ov::element::f16);
    convert_else_op->set_convert_element_type(ov::element::f16);
    if_op->validate_and_infer_types();
    EXPECT_EQ(then_op_res->get_element_type(), ov::element::f16);
    EXPECT_EQ(else_op_res->get_element_type(), ov::element::f32);
}

TEST(type_prop, if_non_const_condition_test) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Xe, Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    Shape out0_shape{32, 40, 10};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
}

TEST(type_prop, if_clone_test) {
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xnew = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ynew = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt, Yt});
    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto new_if = ov::as_type_ptr<op::v8::If>(if_op->clone_with_new_inputs(OutputVector{cond, Xnew, Ynew}));
    EXPECT_EQ(true, true);
}

TEST(type_prop, if_multiple_outputs) {
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xnew = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ynew = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_body_res_1 = make_shared<ov::op::v0::Result>(then_op);
    auto then_body_res_2 = make_shared<ov::op::v0::Result>(Xt);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res_1, then_body_res_2}, ParameterVector{Xt, Yt});
    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_const =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{1, 1, 1}, std::vector<float>{0.5f});
    auto else_body_res_1 = make_shared<ov::op::v0::Result>(else_op);
    auto else_body_res_2 = make_shared<ov::op::v0::Result>(else_const);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res_1, else_body_res_2}, ParameterVector{Xe, Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res1 = if_op->set_output(then_body_res_1, else_body_res_1);
    auto res2 = if_op->set_output(then_body_res_2, else_body_res_2);
    auto result1 = make_shared<ov::op::v0::Result>(res1);
    auto result2 = make_shared<ov::op::v0::Result>(res2);
    Shape out0_shape{32, 40, 10};
    auto sh = result1->get_output_shape(0);
    auto is_dynamic = result2->is_dynamic();
    EXPECT_EQ(out0_shape, sh);
    EXPECT_EQ(is_dynamic, true);
}

TEST(type_prop, if_scalar_condition) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{32, 40, 10});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Xe, Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    Shape out0_shape{32, 40, 10};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
}

TEST(type_prop, if_dynamic_output) {
    // That which we iterate over
    auto X_shape = Shape{1, 20, 5, 30};
    auto Y_shape = Shape{18, 16, 14, 12};
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, X_shape);
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Y_shape);
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Xt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt});

    auto else_op = std::make_shared<op::v1::Maximum>(Ye, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, nullptr);
    if_op->set_input(Y, nullptr, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    auto dynamic_shape = result0->get_output_partial_shape(0);

    EXPECT_EQ(X_shape.size(), dynamic_shape.rank().get_length());
    for (size_t shape_index = 0; shape_index < X_shape.size(); shape_index++) {
        auto x_shape_it = X_shape.begin();
        auto y_shape_it = Y_shape.begin();
        auto res_it = dynamic_shape.begin();
        EXPECT_EQ(std::max(*x_shape_it, *y_shape_it), (*res_it).get_max_length());
        EXPECT_EQ(std::min(*x_shape_it, *y_shape_it), (*res_it).get_min_length());
        x_shape_it++;
        y_shape_it++;
        res_it++;
    }
}

TEST(type_prop, if_dynamic_inputs) {
    // That which we iterate over
    auto X_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic()};
    auto Y_shape = PartialShape{Dimension::dynamic(), 20, 30};
    ;
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, X_shape);
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Y_shape);
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{1});

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Multiply>(Xe, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Xe, Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    auto dynamic_shape = result0->get_output_partial_shape(0);
    auto expected_result = PartialShape{Dimension::dynamic(), 20, 30};
    EXPECT_EQ(3, dynamic_shape.rank().get_length());
    for (auto dim_index = 0; dim_index < 3; dim_index++) {
        auto exp_res_it = expected_result.begin();
        auto res_it = dynamic_shape.begin();
        EXPECT_EQ(*exp_res_it, *res_it);
    }
}

TEST(type_prop, if_scalar_and_1d_union) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic(1));
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});

    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Xt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt});

    auto else_op = std::make_shared<op::v1::Maximum>(Ye, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, nullptr);
    if_op->set_input(Y, nullptr, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    PartialShape out_shape{PartialShape::dynamic(1)};
    auto sh = result0->get_output_partial_shape(0);
    EXPECT_EQ(sh, out_shape);
}

TEST(type_prop, if_scalar_and_1d_static_union) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{8});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});

    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Xt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt});

    auto else_op = std::make_shared<op::v1::Maximum>(Ye, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, nullptr);
    if_op->set_input(Y, nullptr, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    PartialShape out_shape{PartialShape::dynamic(1)};
    auto sh = result0->get_output_partial_shape(0);
    EXPECT_EQ(sh, out_shape);
}

TEST(type_prop, if_output_one_element) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1});
    auto cond = make_shared<ov::op::v0::Parameter>(element::boolean, Shape{});

    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Xt);
    auto then_body_res = make_shared<ov::op::v0::Result>(then_op);
    auto then_body = make_shared<ov::Model>(OutputVector{then_body_res}, ParameterVector{Xt});

    auto else_op = std::make_shared<op::v1::Maximum>(Ye, Ye);
    auto else_body_res = make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_body_res}, ParameterVector{Ye});

    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, nullptr);
    if_op->set_input(Y, nullptr, Ye);
    auto res = if_op->set_output(then_body_res, else_body_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    PartialShape out_shape{1};
    auto sh = result0->get_output_partial_shape(0);
    EXPECT_EQ(sh, out_shape);
}

TEST(type_prop, if_element_type_dynamic) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 40, 10});
    auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, false);

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    // Body
    auto then_op = std::make_shared<op::v1::Add>(Xt, Yt);
    auto then_op_res = std::make_shared<ov::op::v0::Result>(then_op);

    auto then_body = make_shared<ov::Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt});

    auto else_op = std::make_shared<op::v1::Maximum>(Xe, Ye);
    auto else_op_res = std::make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_op_res, else_op_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    Shape out0_shape{32, 40, 10};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
    // Check that If validation when condition is constant validates single body
    if_op->validate_and_infer_types();
    EXPECT_EQ(then_op_res->get_element_type(), ov::element::dynamic);
    EXPECT_EQ(else_op_res->get_element_type(), ov::element::f16);
}

TEST(type_prop, if_invalid_false_body) {
    // That which we iterate over
    auto X = make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 40, 10});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f16, Shape{32, 40});
    auto cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, false);

    // Set up the cell body, a function from (Xi, Yi) -> (Zo)
    // Body parameters
    auto Xt = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::dynamic, PartialShape::dynamic());
    // Body
    auto axes_4d = ov::op::v0::Constant::create(element::i32, ov::Shape{2}, {2, 3});
    auto then_reduce_op = std::make_shared<op::v1::ReduceMean>(Xt, Yt);
    auto then_op = std::make_shared<op::v1::Add>(then_reduce_op, Yt);
    auto then_op_res = std::make_shared<ov::op::v0::Result>(then_op);

    auto then_body = make_shared<ov::Model>(OutputVector{then_op_res}, ParameterVector{Xt, Yt});

    auto axes_3d = ov::op::v0::Constant::create(element::i32, ov::Shape{1}, {2});
    auto else_reduce_op = std::make_shared<op::v1::ReduceMean>(Xe, axes_3d);
    auto else_op = std::make_shared<op::v1::Add>(else_reduce_op, Ye);
    auto else_op_res = std::make_shared<ov::op::v0::Result>(else_op);
    auto else_body = make_shared<ov::Model>(OutputVector{else_op_res}, ParameterVector{Xe, Ye});
    auto if_op = make_shared<op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, Ye);
    auto res = if_op->set_output(then_op_res, else_op_res);
    auto result0 = make_shared<ov::op::v0::Result>(res);
    Shape out0_shape{32, 40};
    auto sh = result0->get_output_shape(0);
    EXPECT_EQ(sh, out0_shape);
    // Check that If validation when condition is constant validates single body
    if_op->validate_and_infer_types();
    EXPECT_EQ(then_op_res->get_element_type(), ov::element::dynamic);
    EXPECT_EQ(else_op_res->get_element_type(), ov::element::f16);
}
