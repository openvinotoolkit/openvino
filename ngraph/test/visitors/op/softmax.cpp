// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/softmax.hpp"

#include "gtest/gtest.h"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, softmax_op) {
    NodeBuilder::get_ops().register_factory<op::v1::Softmax>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto softmax = make_shared<op::v1::Softmax>(data, axis);
    NodeBuilder builder(softmax);
    auto g_softmax = as_type_ptr<op::v1::Softmax>(builder.create());

    EXPECT_EQ(g_softmax->get_axis(), softmax->get_axis());
}
