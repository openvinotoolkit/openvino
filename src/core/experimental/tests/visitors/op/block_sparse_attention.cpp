// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/block_sparse_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, block_sparse_attention_4_input) {
    using ov::op::experimental::BlockSparseAttention;
    NodeBuilder::opset().insert<BlockSparseAttention>();

    const auto query = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 8, 64});
    const auto key = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 2, 64});
    const auto value = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 2, 64});
    const auto block_mask = std::make_shared<Parameter>(ov::element::boolean, PartialShape{1, 1, 2, 2});

    const auto op = std::make_shared<BlockSparseAttention>(query,
                                                           key,
                                                           value,
                                                           block_mask,
                                                           /*block_size_q=*/128,
                                                           /*block_size_kv=*/128,
                                                           /*causal=*/true);

    NodeBuilder builder(op, {query, key, value, block_mask});
    auto g_op = ov::as_type_ptr<BlockSparseAttention>(builder.create());

    EXPECT_EQ(g_op->get_causal(), op->get_causal());
    EXPECT_EQ(g_op->get_block_size_q(), op->get_block_size_q());
    EXPECT_EQ(g_op->get_block_size_kv(), op->get_block_size_kv());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, block_sparse_attention_5_input) {
    using ov::op::experimental::BlockSparseAttention;
    NodeBuilder::opset().insert<BlockSparseAttention>();

    const auto query = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 8, 64});
    const auto key = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 2, 64});
    const auto value = std::make_shared<Parameter>(ov::element::f16, PartialShape{1, 256, 2, 64});
    const auto block_mask = std::make_shared<Parameter>(ov::element::boolean, PartialShape{1, 1, 2, 2});
    const auto scale = std::make_shared<Parameter>(ov::element::f16, PartialShape{});

    const auto op = std::make_shared<BlockSparseAttention>(query,
                                                           key,
                                                           value,
                                                           block_mask,
                                                           scale,
                                                           /*block_size_q=*/128,
                                                           /*block_size_kv=*/128,
                                                           /*causal=*/false);

    NodeBuilder builder(op, {query, key, value, block_mask, scale});
    auto g_op = ov::as_type_ptr<BlockSparseAttention>(builder.create());

    EXPECT_EQ(g_op->get_causal(), op->get_causal());
    EXPECT_EQ(g_op->get_block_size_q(), op->get_block_size_q());
    EXPECT_EQ(g_op->get_block_size_kv(), op->get_block_size_kv());
    EXPECT_EQ(g_op->get_input_size(), 5u);
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
