// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/one_hot.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, one_hot_op) {
    NodeBuilder::get_ops().register_factory<op::v1::OneHot>();
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = op::Constant::create(element::f32, Shape{}, {0.0f});

    int64_t axis = 3;

    auto one_hot = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);
    NodeBuilder builder(one_hot);
    auto g_one_hot = as_type_ptr<op::v1::OneHot>(builder.create());

    EXPECT_EQ(g_one_hot->get_axis(), one_hot->get_axis());
}
