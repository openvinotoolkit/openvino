// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/graph_comparator.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using namespace ngraph::opset8;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, if_op) {
    NodeBuilder::get_ops().register_factory<If>();
    auto X = make_shared<Parameter>(element::f32, Shape{1, 2, 2});
    auto Y = make_shared<Parameter>(element::f32, Shape{1, 2, 2});
    auto cond = std::make_shared<Constant>(element::boolean, Shape{1}, true);
    auto cond2 = std::make_shared<Constant>(element::boolean, Shape{1}, false);
    auto Xt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Yt = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Xe = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto Ye = make_shared<op::Parameter>(element::f32, PartialShape::dynamic());
    auto then_op = std::make_shared<op::v1::Multiply>(Xt, Yt);
    auto res0 = make_shared<op::Result>(then_op);
    auto res1 = make_shared<op::Result>(Xe);
    auto then_body = make_shared<ngraph::Function>(OutputVector{res0}, ParameterVector{Xt, Yt});
    auto else_body = make_shared<ngraph::Function>(OutputVector{res1}, ParameterVector{Xe});
    auto if_op = make_shared<If>(cond);
    if_op->set_then_body(then_body);
    if_op->set_else_body(else_body);
    if_op->set_input(X, Xt, Xe);
    if_op->set_input(Y, Yt, nullptr);
    if_op->set_output(res0, res1);
    if_op->validate_and_infer_types();
    NodeBuilder builder(if_op);
    auto g_if = ov::as_type_ptr<If>(builder.create());
    EXPECT_EQ(g_if->get_input_descriptions(0), if_op->get_input_descriptions(0));
    EXPECT_EQ(g_if->get_input_descriptions(1), if_op->get_input_descriptions(1));
    EXPECT_EQ(g_if->get_output_descriptions(0), if_op->get_output_descriptions(0));
    EXPECT_EQ(g_if->get_output_descriptions(1), if_op->get_output_descriptions(1));
    EXPECT_TRUE(compare_functions(g_if->get_then_body(), if_op->get_then_body()).first);
    EXPECT_TRUE(compare_functions(g_if->get_else_body(), if_op->get_else_body()).first);
}
