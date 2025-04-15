// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, if_op) {
    NodeBuilder::opset().insert<ov::op::v8::If>();
    auto X = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    auto Y = make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 2});
    auto cond = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, true);
    auto cond2 = std::make_shared<ov::op::v0::Constant>(element::boolean, Shape{1}, false);
    auto Xt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape::dynamic());
    auto then_op = std::make_shared<op::v1::Multiply>(Xt, Yt);
    auto res0 = make_shared<ov::op::v0::Result>(then_op);
    auto res1 = make_shared<ov::op::v0::Result>(Xe);
    auto then_body = make_shared<ov::Model>(OutputVector{res0}, ParameterVector{Xt, Yt});
    auto else_body = make_shared<ov::Model>(OutputVector{res1}, ParameterVector{Xe});
    auto if_op = make_shared<ov::op::v8::If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_output(res0, res1);
    if_op->validate_and_infer_types();
    NodeBuilder builder(if_op);
    auto g_if = ov::as_type_ptr<ov::op::v8::If>(builder.create());
    EXPECT_EQ(g_if->get_input_descriptions(0), if_op->get_input_descriptions(0));
    EXPECT_EQ(g_if->get_input_descriptions(1), if_op->get_input_descriptions(1));
    EXPECT_EQ(g_if->get_output_descriptions(0), if_op->get_output_descriptions(0));
    EXPECT_EQ(g_if->get_output_descriptions(1), if_op->get_output_descriptions(1));
    EXPECT_TRUE(compare_functions(g_if->get_then_body(), if_op->get_then_body()).first);
    EXPECT_TRUE(compare_functions(g_if->get_else_body(), if_op->get_else_body()).first);
}
