// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {
using ov::op::internal::QSAIndexer, ov::op::v0::Parameter, ov::test::NodeBuilder;

TEST(attributes, qsa_indexer_internal) {
    NodeBuilder::opset().insert<QSAIndexer>();
    const OutputVector inputs{std::make_shared<Parameter>(element::f32, Shape{1, 8, 4, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 1, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 64}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 64}),
                              std::make_shared<Parameter>(element::i32, Shape{1}),
                              std::make_shared<Parameter>(element::f32, Shape{128}),
                              std::make_shared<Parameter>(element::f32, Shape{128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 1, 2, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 1, 3, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 128})};
    const auto op =
        std::make_shared<QSAIndexer>(inputs, QSAIndexer::Config{4, 16, 1, 64, 128, 2e-6f, 0.5f, false, true});
    NodeBuilder builder(op, inputs);
    const auto g_op = ov::as_type_ptr<QSAIndexer>(builder.create());

    constexpr auto expected_attr_count = 9;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    const auto& expected = op->get_config();
    const auto& actual = g_op->get_config();
    EXPECT_EQ(actual.key_block_size, expected.key_block_size);
    EXPECT_EQ(actual.block_topk, expected.block_topk);
    EXPECT_EQ(actual.num_selection_heads, expected.num_selection_heads);
    EXPECT_EQ(actual.indexer_rotary_dim, expected.indexer_rotary_dim);
    EXPECT_EQ(actual.indexer_head_dim, expected.indexer_head_dim);
    EXPECT_EQ(actual.epsilon, expected.epsilon);
    EXPECT_EQ(actual.scale, expected.scale);
    EXPECT_EQ(actual.query_norm, expected.query_norm);
    EXPECT_EQ(actual.query_rope, expected.query_rope);
}
}  // namespace ov::test
