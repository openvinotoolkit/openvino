// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention_indexer.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::op::v17::SparseAttentionIndexer, ov::test::NodeBuilder;

TEST(attributes, sparse_attention_indexer_v17) {
    NodeBuilder::opset().insert<SparseAttentionIndexer>();
    const auto index_query = std::make_shared<Parameter>(element::f32, Shape{1, 8, 7, 128});
    const auto block_index_keys = std::make_shared<Parameter>(element::f32, Shape{1, 100, 128});
    const auto key_length = std::make_shared<Parameter>(element::i32, Shape{1});

    const auto op = std::make_shared<SparseAttentionIndexer>(index_query, block_index_keys, key_length, 4, 64, 2);
    NodeBuilder builder(op, {index_query, block_index_keys, key_length});
    const auto g_op = ov::as_type_ptr<SparseAttentionIndexer>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_key_block_size(), op->get_key_block_size());
    EXPECT_EQ(g_op->get_token_budget(), op->get_token_budget());
    EXPECT_EQ(g_op->get_num_selection_heads(), op->get_num_selection_heads());
}
}  // namespace ov::test
