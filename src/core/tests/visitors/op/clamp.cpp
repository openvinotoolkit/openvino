// Copyright (C) 2018-2022 Intel Corporation
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

TEST(attributes, clamp_op) {
    NodeBuilder::get_ops().register_factory<opset1::Clamp>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    double min = 0.4;
    double max = 5.6;

    const auto clamp = make_shared<opset1::Clamp>(data, min, max);
    NodeBuilder builder(clamp);
    auto g_clamp = ov::as_type_ptr<opset1::Clamp>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_clamp->get_min(), clamp->get_min());
    EXPECT_EQ(g_clamp->get_max(), clamp->get_max());
}
