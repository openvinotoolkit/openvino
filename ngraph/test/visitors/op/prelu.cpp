// Copyright (C) 2018-2021 Intel Corporation
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

TEST(attributes, prelu_op)
{
    NodeBuilder::get_ops().register_factory<opset1::PRelu>();
    auto x1 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto prelu = make_shared<opset1::PRelu>(x1, x2);
    NodeBuilder builder(prelu);
    auto g_prelu = as_type_ptr<opset1::PRelu>(builder.create());

    EXPECT_EQ(g_prelu->get_autob(), prelu->get_autob());
}
