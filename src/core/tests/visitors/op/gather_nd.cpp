// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_nd.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, gather_nd_v5_op) {
    NodeBuilder::opset().insert<ov::op::v5::GatherND>();
    int batch_dims = 1;
    auto P = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 1});
    auto G = make_shared<op::v5::GatherND>(P, I, batch_dims);

    NodeBuilder builder(G, {P, I});
    auto g_G = ov::as_type_ptr<ov::op::v5::GatherND>(builder.create());

    EXPECT_EQ(g_G->get_batch_dims(), G->get_batch_dims());
}

TEST(attributes, gather_nd_v8_op) {
    NodeBuilder::opset().insert<ov::op::v8::GatherND>();
    int batch_dims = 1;
    auto P = make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3, 4});
    auto I = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 1});
    auto G = make_shared<op::v8::GatherND>(P, I, batch_dims);

    NodeBuilder builder(G, {P, I});
    auto g_G = ov::as_type_ptr<ov::op::v8::GatherND>(builder.create());

    EXPECT_EQ(g_G->get_batch_dims(), G->get_batch_dims());
}
