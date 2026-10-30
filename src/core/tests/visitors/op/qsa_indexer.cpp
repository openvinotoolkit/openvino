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
    const OutputVector inputs{std::make_shared<Parameter>(element::f32, Shape{1, 4, 8, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 64}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 8, 64}),
                              std::make_shared<Parameter>(element::i32, Shape{1}),
                              std::make_shared<Parameter>(element::f32, Shape{128}),
                              std::make_shared<Parameter>(element::f32, Shape{128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 2, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 3, 128}),
                              std::make_shared<Parameter>(element::f32, Shape{1, 128})};
    const auto op = std::make_shared<QSAIndexer>(inputs, QSAIndexer::Config{4, 16, 1, 2e-6f, 3e-6f, true});
    NodeBuilder builder(op, inputs);
    const auto g_op = ov::as_type_ptr<QSAIndexer>(builder.create());

    constexpr auto expected_attr_count = 6;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_config().key_block_size, op->get_config().key_block_size);
    EXPECT_EQ(g_op->get_config().token_budget, op->get_config().token_budget);
    EXPECT_EQ(g_op->get_config().num_selection_heads, op->get_config().num_selection_heads);
    EXPECT_EQ(g_op->get_config().q_epsilon, op->get_config().q_epsilon);
    EXPECT_EQ(g_op->get_config().k_epsilon, op->get_config().k_epsilon);
    EXPECT_EQ(g_op->get_config().is_causal, op->get_config().is_causal);
}
}  // namespace ov::test
