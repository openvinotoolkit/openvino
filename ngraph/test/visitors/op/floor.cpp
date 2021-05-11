// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, floor_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Mod>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{5, 2});

    auto floor = make_shared<opset1::Floor>(A);
    NodeBuilder builder(floor);
    auto g_floor = as_type_ptr<opset1::Floor>(builder.create());

    EXPECT_EQ(g_floor->get_autob(), floor->get_autob());
}
