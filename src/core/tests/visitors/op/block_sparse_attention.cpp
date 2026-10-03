// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/block_sparse_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::v0::Parameter, ov::test::NodeBuilder;

TEST(attributes, block_sparse_attention_v17) {
    NodeBuilder::opset().insert<ov::op::v17::BlockSparseAttention>();
    const auto query = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 4, 8, 16});
    const auto key = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 4, 32, 16});
    const auto value = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{2, 4, 32, 16});
    const auto block_indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{2, 4, 2, 3});

    const auto op = std::make_shared<ov::op::v17::BlockSparseAttention>(query, key, value, block_indices, 4, true);
    NodeBuilder builder(op, {query, key, value, block_indices});
    auto g_op = ov::as_type_ptr<ov::op::v17::BlockSparseAttention>(builder.create());

    EXPECT_EQ(g_op->get_block_size(), op->get_block_size());
    EXPECT_EQ(g_op->get_causal(), op->get_causal());
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
}

}  // namespace ov::test
