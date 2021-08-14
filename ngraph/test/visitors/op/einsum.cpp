// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/op/einsum.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, einsum_v7_op) {
    NodeBuilder::get_ops().register_factory<op::v7::Einsum>();
    auto input1 = make_shared<op::Parameter>(element::i32, Shape{2, 3});
    auto input2 = make_shared<op::Parameter>(element::i32, Shape{3, 4});
    std::string equation = "ab,bc->ac";
    auto einsum = make_shared<op::v7::Einsum>(OutputVector{input1, input2}, equation);
    NodeBuilder builder(einsum);
    auto g_einsum = as_type_ptr<op::v7::Einsum>(builder.create());
    EXPECT_EQ(g_einsum->get_equation(), einsum->get_equation());
}
