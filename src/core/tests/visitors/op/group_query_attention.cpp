// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, group_query_attention_attrs) {
    using ov::op::internal::GroupQueryAttention;
    using ov::op::internal::GroupQueryAttentionQuantType;
    NodeBuilder::opset().insert<GroupQueryAttention>();

    const auto f32 = ov::element::f32;
    const ov::OutputVector args{std::make_shared<Parameter>(f32, PartialShape{1, 4, -1, 16}),
                                std::make_shared<Parameter>(f32, PartialShape{1, 2, -1, 16}),
                                std::make_shared<Parameter>(f32, PartialShape{1, 2, -1, 16}),
                                std::make_shared<Parameter>(f32, PartialShape{1, 2, -1, 16}),
                                std::make_shared<Parameter>(f32, PartialShape{1, 2, -1, 16}),
                                std::make_shared<Parameter>(ov::element::i32, PartialShape{1}),
                                std::make_shared<Parameter>(ov::element::i32, PartialShape{})};
    const auto op = std::make_shared<GroupQueryAttention>(args,
                                                          4,
                                                          2,
                                                          0.125f,
                                                          false,
                                                          false,
                                                          /*kv_cache_bit_width*/ 0,
                                                          GroupQueryAttentionQuantType::NONE,
                                                          GroupQueryAttentionQuantType::NONE,
                                                          /*local_window_size*/ 8,
                                                          /*sliding_window_cache*/ false,
                                                          /*smooth_softmax*/ true,
                                                          /*causal*/ true,
                                                          /*softcap*/ 50.0f);

    NodeBuilder builder(op, args);
    const auto g_op = ov::as_type_ptr<GroupQueryAttention>(builder.create());
    ASSERT_NE(g_op, nullptr);
    EXPECT_EQ(builder.get_value_map_size(), 13);
    EXPECT_EQ(g_op->get_num_heads(), op->get_num_heads());
    EXPECT_EQ(g_op->get_kv_num_heads(), op->get_kv_num_heads());
    EXPECT_EQ(g_op->get_scale(), op->get_scale());
    EXPECT_EQ(g_op->get_local_window_size(), op->get_local_window_size());
    EXPECT_EQ(g_op->get_smooth_softmax(), op->get_smooth_softmax());
    EXPECT_EQ(g_op->get_causal(), op->get_causal());
    EXPECT_EQ(g_op->get_softcap(), op->get_softcap());
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
