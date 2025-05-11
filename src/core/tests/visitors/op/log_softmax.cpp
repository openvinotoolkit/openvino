// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, logsoftmax_op) {
    NodeBuilder::opset().insert<ov::op::v5::LogSoftmax>();
    auto data = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 2, 3});

    int64_t axis = 2;

    const auto logsoftmax = make_shared<ov::op::v5::LogSoftmax>(data, axis);
    NodeBuilder builder(logsoftmax, {data});
    auto g_logsoftmax = ov::as_type_ptr<ov::op::v5::LogSoftmax>(builder.create());

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_logsoftmax->get_axis(), logsoftmax->get_axis());
}
