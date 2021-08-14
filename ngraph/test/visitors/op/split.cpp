// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/split.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, split_op) {
    NodeBuilder::get_ops().register_factory<op::v1::Split>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{});
    auto num_splits = 2;
    auto split = make_shared<op::v1::Split>(data, axis, num_splits);
    NodeBuilder builder(split);
    auto g_split = as_type_ptr<op::v1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}
