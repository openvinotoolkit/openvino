// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/*
#include "binary_ops.hpp"
#include "ngraph/opsets/opset1.hpp"

using Type = ::testing::Types<BinaryOperatorType<ngraph::opset1::Divide, ngraph::element::f32>>;

INSTANTIATE_TYPED_TEST_SUITE_P(visitor_with_auto_broadcast, BinaryOperatorVisitor, Type, BinaryOperatorTypeName);

*/
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, divide) {
    NodeBuilder::get_ops().register_factory<opset1::Divide>();

    const auto in1 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    const auto in2 = make_shared<op::Parameter>(element::f32, Shape{2, 4});
    const bool pythondiv = true;
    const op::AutoBroadcastSpec& auto_broadcast = op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY);
    const auto divide = make_shared<opset1::Divide>(in1, in2, pythondiv, auto_broadcast);

    NodeBuilder builder(divide);
    auto g_divide = ov::as_type_ptr<opset1::Divide>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_divide->is_pythondiv(), divide->is_pythondiv());
    EXPECT_EQ(g_divide->get_autob(), divide->get_autob());
}
