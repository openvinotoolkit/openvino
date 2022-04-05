// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset7.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, roll_op) {
    NodeBuilder::get_ops().register_factory<opset7::Roll>();
    const auto A = make_shared<op::Parameter>(element::f32, Shape{4, 3});
    const auto B = make_shared<op::Constant>(element::i32, Shape{3});
    const auto C = make_shared<op::Constant>(element::i32, Shape{3});

    const auto roll = make_shared<opset7::Roll>(A, B, C);
    NodeBuilder builder(roll);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
