// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_gather.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::op::internal::TPGather;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, tp_gather) {
    NodeBuilder::opset().insert<TPGather>();
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{2, 8});
    const auto op = std::make_shared<TPGather>(data, 0, 4, 1, 0xFEDCBA9876543210ULL, 0);

    NodeBuilder builder(op, {data});
    const auto g_op = ov::as_type_ptr<TPGather>(builder.create());

    constexpr auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_rank(), op->get_rank());
    EXPECT_EQ(g_op->get_world_size(), op->get_world_size());
    EXPECT_EQ(g_op->get_group_id(), op->get_group_id());
    EXPECT_EQ(g_op->get_collective_id(), op->get_collective_id());
    EXPECT_EQ(g_op->get_axis(), op->get_axis());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
