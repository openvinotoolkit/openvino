// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_tree.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, gather_tree_op) {
    NodeBuilder::opset().insert<ov::op::v1::GatherTree>();

    auto step_ids = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto parent_idx = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 3});
    auto max_seq_len = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto end_token = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{});

    auto gather_tree = std::make_shared<ov::op::v1::GatherTree>(step_ids, parent_idx, max_seq_len, end_token);
    NodeBuilder builder(gather_tree, {step_ids, parent_idx, max_seq_len, end_token});
    EXPECT_NO_THROW(auto g_gather_tree = ov::as_type_ptr<ov::op::v1::GatherTree>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
