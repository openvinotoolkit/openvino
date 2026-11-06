// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
TEST(attributes, paged_qsa_indexer_internal) {
    using ov::op::internal::PagedQSAIndexer;
    NodeBuilder::opset().insert<PagedQSAIndexer>();
    const std::vector<PartialShape> shapes{{6, 64},
                                           {6, 16},
                                           {6, 8},
                                           {6, 8},
                                           {16},
                                           {16},
                                           {2},
                                           {3},
                                           {10},
                                           {3},
                                           {5, 1, 16, 16},
                                           {5, 4, 16},
                                           {5, 1, 4, 16}};
    OutputVector inputs;
    for (size_t idx = 0; idx < shapes.size(); ++idx) {
        inputs.push_back(std::make_shared<op::v0::Parameter>(
            idx >= PagedQSAIndexer::InputIdx::PAST_LENS && idx <= PagedQSAIndexer::InputIdx::BLOCK_INDICES_BEGINS
                ? element::i32
                : element::f32,
            shapes[idx]));
    }
    for (const bool query_norm : {false, true}) {
        for (const bool query_rope : {false, true}) {
            const auto op = std::make_shared<PagedQSAIndexer>(
                inputs,
                PagedQSAIndexer::Config{4, 2, 1, 8, 16, 2e-6f, 0.5f, query_norm, query_rope});
            NodeBuilder builder(op, inputs);
            const auto restored = ov::as_type_ptr<PagedQSAIndexer>(builder.create());
            ASSERT_NE(restored, nullptr);
            EXPECT_EQ(builder.get_value_map_size(), 9);
            EXPECT_EQ(restored->get_config().key_block_size, op->get_config().key_block_size);
            EXPECT_EQ(restored->get_config().block_topk, op->get_config().block_topk);
            EXPECT_EQ(restored->get_config().num_selection_heads, op->get_config().num_selection_heads);
            EXPECT_EQ(restored->get_config().indexer_rotary_dim, op->get_config().indexer_rotary_dim);
            EXPECT_EQ(restored->get_config().indexer_head_dim, op->get_config().indexer_head_dim);
            EXPECT_FLOAT_EQ(restored->get_config().epsilon, op->get_config().epsilon);
            EXPECT_FLOAT_EQ(restored->get_config().scale, op->get_config().scale);
            EXPECT_EQ(restored->get_config().query_norm, query_norm);
            EXPECT_EQ(restored->get_config().query_rope, query_rope);
        }
    }
}
}  // namespace ov::test
