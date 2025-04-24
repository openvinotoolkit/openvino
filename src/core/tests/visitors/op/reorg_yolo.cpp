// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reorg_yolo.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, reorg_yolo_op_stride) {
    NodeBuilder::opset().insert<ov::op::v0::ReorgYolo>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, 2);
    NodeBuilder builder(op, {data});
    const auto g_op = ov::as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}

TEST(attributes, reorg_yolo_op_strides) {
    NodeBuilder::opset().insert<ov::op::v0::ReorgYolo>();
    const auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{1, 64, 26, 26});

    const auto op = make_shared<op::v0::ReorgYolo>(data, Strides{2});
    NodeBuilder builder(op, {data});
    const auto g_op = ov::as_type_ptr<op::v0::ReorgYolo>(builder.create());

    EXPECT_EQ(g_op->get_strides(), op->get_strides());
}
