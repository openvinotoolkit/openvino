// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/tanh.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, tanh_op) {
    NodeBuilder::opset().insert<op::v0::Tanh>();
    const auto data_node = make_shared<op::v0::Parameter>(element::f32, Shape{1});
    const auto tanh = make_shared<op::v0::Tanh>(data_node);

    const NodeBuilder builder(tanh);
    const auto tanh_attr_number = 0;

    EXPECT_EQ(builder.get_value_map_size(), tanh_attr_number);
}
