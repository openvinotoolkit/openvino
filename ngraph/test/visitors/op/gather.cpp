// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset7.hpp"

#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_v7_op)
{
    NodeBuilder::get_ops().register_factory<opset7::Gather>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    auto axis = make_shared<opset1::Constant>(element::i32, Shape{}, 2);
    int64_t batch_dims = 1;

    auto gather = make_shared<opset7::Gather>(data, indices, axis, batch_dims);
    NodeBuilder builder(gather);
    auto g_gather = as_type_ptr<opset7::Gather>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}
