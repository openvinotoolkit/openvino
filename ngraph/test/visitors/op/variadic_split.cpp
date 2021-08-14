// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/variadic_split.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, variadic_split_op) {
    NodeBuilder::get_ops().register_factory<op::v1::VariadicSplit>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{1});
    auto split_lengths = make_shared<op::Parameter>(element::i32, Shape{1});

    auto split = make_shared<op::v1::VariadicSplit>(data, axis, split_lengths);
    NodeBuilder builder(split);
    const auto expected_attr_count = 0;

    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
