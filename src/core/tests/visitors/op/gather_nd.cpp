// Copyright (C) 2018-2022 Intel Corporation
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
    int batch_dims = 1;
    auto P = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto I = make_shared<op::Parameter>(element::i32, Shape{2, 1});
    auto G = make_shared<op::v5::GatherND>(P, I, batch_dims);

    NodeBuilder builder(G);
    auto g_G = ov::as_type_ptr<opset5::GatherND>(builder.create());

    EXPECT_EQ(g_G->get_batch_dims(), G->get_batch_dims());
}

TEST(attributes, gather_nd_v8_op) {
    NodeBuilder::get_ops().register_factory<opset8::GatherND>();
    int batch_dims = 1;
    auto P = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4});
    auto I = make_shared<op::Parameter>(element::i32, Shape{2, 1});
    auto G = make_shared<op::v8::GatherND>(P, I, batch_dims);

    NodeBuilder builder(G);
    auto g_G = ov::as_type_ptr<opset8::GatherND>(builder.create());

    EXPECT_EQ(g_G->get_batch_dims(), G->get_batch_dims());
}
