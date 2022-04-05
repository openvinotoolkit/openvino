// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, mvn_v1_op) {
    NodeBuilder::get_ops().register_factory<opset3::MVN>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});

    const auto axes = AxisSet{0, 1};

    const auto op = make_shared<opset3::MVN>(data, true, false, 0.1);
    op->set_reduction_axes(axes);
    NodeBuilder builder(op);
    const auto g_op = ov::as_type_ptr<opset3::MVN>(builder.create());
    const auto expected_attr_count = 4;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_reduction_axes(), op->get_reduction_axes());
    EXPECT_EQ(g_op->get_across_channels(), op->get_across_channels());
    EXPECT_EQ(g_op->get_normalize_variance(), op->get_normalize_variance());
    EXPECT_EQ(g_op->get_eps(), op->get_eps());
}

TEST(attributes, mvn_v6_op) {
    NodeBuilder::get_ops().register_factory<opset6::MVN>();
    const auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto axes = ngraph::opset6::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {2, 3});

    const auto op = make_shared<opset6::MVN>(data, axes, false, 0.1, op::MVNEpsMode::INSIDE_SQRT);

    NodeBuilder builder(op);
    const auto g_op = ov::as_type_ptr<opset6::MVN>(builder.create());
    const auto expected_attr_count = 3;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_eps_mode(), op->get_eps_mode());
    EXPECT_EQ(g_op->get_normalize_variance(), op->get_normalize_variance());
    EXPECT_EQ(g_op->get_eps(), op->get_eps());
}
