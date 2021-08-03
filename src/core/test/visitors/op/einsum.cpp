// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset7.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, einsum_v7_op)
{
    NodeBuilder::get_ops().register_factory<opset7::Einsum>();
    auto input1 = make_shared<opset1::Parameter>(element::i32, Shape{2, 3});
    auto input2 = make_shared<opset1::Parameter>(element::i32, Shape{3, 4});
    std::string equation = "ab,bc->ac";
    auto einsum = make_shared<opset7::Einsum>(OutputVector{input1, input2}, equation);
    NodeBuilder builder(einsum);
    auto g_einsum = as_type_ptr<opset7::Einsum>(builder.create());
    EXPECT_EQ(g_einsum->get_equation(), einsum->get_equation());
}
