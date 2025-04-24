// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/shuffle_channels.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, shuffle_channels_op) {
    using ShuffleChannels = ov::op::v0::ShuffleChannels;

    NodeBuilder::opset().insert<ShuffleChannels>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 64, 16, 16});
    auto axis = 1;
    auto groups = 2;
    auto shuffle_channels = make_shared<ShuffleChannels>(data, axis, groups);
    NodeBuilder builder(shuffle_channels, {data});
    auto g_shuffle_channels = ov::as_type_ptr<ShuffleChannels>(builder.create());

    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_shuffle_channels->get_axis(), shuffle_channels->get_axis());
    EXPECT_EQ(g_shuffle_channels->get_group(), shuffle_channels->get_group());
}
