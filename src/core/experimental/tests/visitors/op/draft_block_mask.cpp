// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/draft_block_mask.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, draft_block_mask_3_input) {
    using ov::op::experimental::DraftBlockMask;
    NodeBuilder::opset().insert<DraftBlockMask>();

    const auto q_windows = std::make_shared<Parameter>(ov::element::f16, PartialShape{4, 128, 4});
    const auto k_windows = std::make_shared<Parameter>(ov::element::f16, PartialShape{4, 128, 4});
    const auto local_mask = std::make_shared<Parameter>(ov::element::boolean, PartialShape{2, 2});

    const auto op = std::make_shared<DraftBlockMask>(q_windows,
                                                     k_windows,
                                                     local_mask,
                                                     /*num_heads=*/2,
                                                     /*seqlen=*/2,
                                                     /*topk=*/3,
                                                     /*batch_size=*/1);

    NodeBuilder builder(op, {q_windows, k_windows, local_mask});
    auto g_op = ov::as_type_ptr<DraftBlockMask>(builder.create());

    EXPECT_EQ(g_op->get_num_heads(), op->get_num_heads());
    EXPECT_EQ(g_op->get_seqlen(), op->get_seqlen());
    EXPECT_EQ(g_op->get_topk(), op->get_topk());
    EXPECT_EQ(g_op->get_batch_size(), op->get_batch_size());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST(attributes, draft_block_mask_5_input) {
    using ov::op::experimental::DraftBlockMask;
    NodeBuilder::opset().insert<DraftBlockMask>();

    const auto q_windows = std::make_shared<Parameter>(ov::element::f16, PartialShape::dynamic(3));
    const auto k_windows = std::make_shared<Parameter>(ov::element::f16, PartialShape::dynamic(3));
    const auto local_mask = std::make_shared<Parameter>(ov::element::boolean, PartialShape::dynamic(2));
    const auto seqlen = std::make_shared<Parameter>(ov::element::i64, PartialShape{1});
    const auto topk = std::make_shared<Parameter>(ov::element::i64, PartialShape{1});

    const auto op = std::make_shared<DraftBlockMask>(q_windows,
                                                     k_windows,
                                                     local_mask,
                                                     seqlen,
                                                     topk,
                                                     /*num_heads=*/2,
                                                     /*batch_size=*/1);

    NodeBuilder builder(op, {q_windows, k_windows, local_mask, seqlen, topk});
    auto g_op = ov::as_type_ptr<DraftBlockMask>(builder.create());

    EXPECT_EQ(g_op->get_num_heads(), op->get_num_heads());
    EXPECT_EQ(g_op->get_batch_size(), op->get_batch_size());
    EXPECT_EQ(g_op->get_input_size(), 5u);
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
