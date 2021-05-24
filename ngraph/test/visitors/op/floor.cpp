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

TEST(attributes, floor_op)
{
    NodeBuilder::get_ops().register_factory<opset1::Floor>();
    const auto A = make_shared<op::Parameter>(element::f32, Shape{5, 2});

    const auto floor = make_shared<opset1::Floor>(A);
    NodeBuilder builder(floor);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
