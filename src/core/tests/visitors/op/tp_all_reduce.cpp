// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/tp_all_reduce.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::op::internal::ReduceOp;
using ov::op::internal::TPAllReduce;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

// An orchestrator may hand out any 64-bit collective id, a hash for instance; the whole of it survives.
TEST(attributes, tp_all_reduce) {
    NodeBuilder::opset().insert<TPAllReduce>();
    constexpr uint64_t collective_id = 0xFEDCBA9876543210ULL;
    const auto data = std::make_shared<Parameter>(ov::element::f16, ov::PartialShape{2, 8});

    for (const auto reduce_op : {ReduceOp::SUM, ReduceOp::MAX, ReduceOp::MIN}) {
        const auto op = std::make_shared<TPAllReduce>(data, 1, 3, 2, collective_id, reduce_op);
        NodeBuilder builder(op, {data});
        const auto g_op = ov::as_type_ptr<TPAllReduce>(builder.create());

        constexpr auto expected_attr_count = 5;
        EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
        EXPECT_EQ(g_op->get_rank(), op->get_rank());
        EXPECT_EQ(g_op->get_world_size(), op->get_world_size());
        EXPECT_EQ(g_op->get_group_id(), op->get_group_id());
        EXPECT_EQ(g_op->get_collective_id(), collective_id);
        EXPECT_EQ(g_op->get_reduce_op(), reduce_op);
        EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
        EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    }
}

// The names are what the IR stores.
TEST(attributes, tp_all_reduce_reduce_op_names) {
    EXPECT_EQ(ov::as_string(ReduceOp::SUM), "sum");
    EXPECT_EQ(ov::as_string(ReduceOp::MAX), "max");
    EXPECT_EQ(ov::as_string(ReduceOp::MIN), "min");
    EXPECT_EQ(ov::as_enum<ReduceOp>("max"), ReduceOp::MAX);
}
