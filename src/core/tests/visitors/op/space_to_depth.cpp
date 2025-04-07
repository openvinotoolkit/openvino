// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, space_to_depth_op) {
    NodeBuilder::opset().insert<ov::op::v0::SpaceToDepth>();
    auto data = make_shared<ov::op::v0::Parameter>(element::i32, Shape{2, 3, 50, 50});

    auto block_size = 2;
    auto mode = ov::op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;

    auto space_to_depth = make_shared<ov::op::v0::SpaceToDepth>(data, mode, block_size);
    NodeBuilder builder(space_to_depth, {data});
    auto g_space_to_depth = ov::as_type_ptr<ov::op::v0::SpaceToDepth>(builder.create());

    // attribute count
    const auto expected_attr_count = 2;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    // space_to_depth attributes
    EXPECT_EQ(g_space_to_depth->get_block_size(), space_to_depth->get_block_size());
    EXPECT_EQ(g_space_to_depth->get_mode(), space_to_depth->get_mode());
}
