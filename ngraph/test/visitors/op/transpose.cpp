// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/transpose.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;

TEST(attributes, transpose_op) {
    NodeBuilder::get_ops().register_factory<op::v1::Transpose>();
    const auto data_input = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto axes_order_input = make_shared<op::Parameter>(element::i32, Shape{3});

    const auto op = make_shared<op::v1::Transpose>(data_input, axes_order_input);

    NodeBuilder builder(op);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
