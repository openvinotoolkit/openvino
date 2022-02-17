// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, select) {
    NodeBuilder::get_ops().register_factory<opset1::Select>();
    auto in_cond = std::make_shared<op::Parameter>(element::boolean, Shape{3, 2});
    auto in_then = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});
    auto in_else = std::make_shared<op::Parameter>(element::f32, Shape{3, 2});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto select = std::make_shared<opset1::Select>(in_cond, in_then, in_else, auto_broadcast);
    NodeBuilder builder(select);

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    auto g_select = ov::as_type_ptr<opset1::Select>(builder.create());
    EXPECT_EQ(g_select->get_autob(), select->get_autob());
}
