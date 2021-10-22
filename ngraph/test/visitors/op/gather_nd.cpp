// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "ngraph/opsets/opset8.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, gather_nd_v5_op) {
    NodeBuilder::get_ops().register_factory<opset5::GatherND>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    int64_t batch_dims = 1;

    auto gather = make_shared<opset5::GatherND>(data, indices, batch_dims);
    NodeBuilder builder(gather);
    auto g_gather = ov::as_type_ptr<opset5::GatherND>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}

TEST(attributes, gather_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::GatherND>();
    auto data = make_shared<opset1::Parameter>(element::i32, Shape{2, 3, 4});
    auto indices = make_shared<opset1::Parameter>(element::i32, Shape{2});
    int64_t batch_dims = 1;

    auto gather = make_shared<opset8::GatherND>(data, indices, batch_dims);
    NodeBuilder builder(gather);
    auto g_gather = ov::as_type_ptr<opset8::GatherND>(builder.create());

    EXPECT_EQ(g_gather->get_batch_dims(), gather->get_batch_dims());
}
