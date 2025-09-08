// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, softmax_op) {
    NodeBuilder::opset().insert<ov::op::v1::Softmax>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto softmax = make_shared<ov::op::v1::Softmax>(data, axis);
    NodeBuilder builder(softmax, {data});
    auto g_softmax = ov::as_type_ptr<ov::op::v1::Softmax>(builder.create());

    EXPECT_EQ(g_softmax->get_axis(), softmax->get_axis());
}
