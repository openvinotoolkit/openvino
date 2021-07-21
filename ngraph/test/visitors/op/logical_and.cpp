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

TEST(attributes, logical_and_op)
{
    NodeBuilder::get_ops().register_factory<opset1::LogicalAnd>();
    auto x1 = make_shared<op::Parameter>(element::boolean, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::boolean, Shape{200});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto logical_and = make_shared<opset1::LogicalAnd>(x1, x2, auto_broadcast);
    NodeBuilder builder(logical_and);
    auto g_logical_and = as_type_ptr<opset1::LogicalAnd>(builder.create());

    EXPECT_EQ(g_logical_and->get_autob(), logical_and->get_autob());
}
