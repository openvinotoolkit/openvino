// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, concat_op) {
    NodeBuilder::get_ops().register_factory<opset1::Concat>();
    auto input1 = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto input2 = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    auto input3 = make_shared<op::Parameter>(element::i64, Shape{1, 2, 3});
    int64_t axis = 2;

    auto concat = make_shared<opset1::Concat>(ov::NodeVector{input1, input2, input3}, axis);
    NodeBuilder builder(concat, {input1, input2, input3});
    auto g_concat = ov::as_type_ptr<opset1::Concat>(builder.create());

    EXPECT_EQ(g_concat->get_axis(), concat->get_axis());
}
