// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::op::v17::SparseAttention, ov::test::NodeBuilder;

TEST(attributes, sparse_attention_v17) {
    NodeBuilder::opset().insert<SparseAttention>();
    const auto query = std::make_shared<Parameter>(element::f32, Shape{2, 24, 8, 64});
    const auto key = std::make_shared<Parameter>(element::f32, Shape{2, 2, 16, 64});
    const auto value = std::make_shared<Parameter>(element::f32, Shape{2, 2, 16, 32});
    const auto block_indices = std::make_shared<Parameter>(element::i32, Shape{2, 1, 2, 4});
    const auto block_count = std::make_shared<Parameter>(element::i32, Shape{2, 1, 2});

    const auto op = std::make_shared<SparseAttention>(query, key, value, block_indices, block_count, 4, 4, false);
    NodeBuilder builder(op, {query, key, value, block_indices, block_count});
    const auto g_op = ov::as_type_ptr<SparseAttention>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_key_block_size(), op->get_key_block_size());
    EXPECT_EQ(g_op->get_query_block_size(), op->get_query_block_size());
    EXPECT_EQ(g_op->get_causal(), op->get_causal());
}
}  // namespace ov::test
