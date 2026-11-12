// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_qsa_indexer.hpp"

#include <gtest/gtest.h>

#include "openvino/op/parameter.hpp"
#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, paged_qsa_indexer) {
    using ov::op::internal::PagedQSAIndexer;
    NodeBuilder::opset().insert<PagedQSAIndexer>();

    const auto f32 = ov::element::f32;
    const auto i32 = ov::element::i32;
    // Six tokens, H_iq = 2, H_ik = 1, D_idx = 4, two rotary dims, eight physical pages of eight slots (r = 4).
    const ov::OutputVector args{std::make_shared<Parameter>(f32, PartialShape{6, 8}),
                                std::make_shared<Parameter>(f32, PartialShape{6, 4}),
                                std::make_shared<Parameter>(f32, PartialShape{6, 2}),
                                std::make_shared<Parameter>(f32, PartialShape{6, 2}),
                                std::make_shared<Parameter>(f32, PartialShape{4}),
                                std::make_shared<Parameter>(f32, PartialShape{4}),
                                std::make_shared<Parameter>(i32, PartialShape{2}),
                                std::make_shared<Parameter>(i32, PartialShape{3}),
                                std::make_shared<Parameter>(i32, PartialShape{4}),
                                std::make_shared<Parameter>(i32, PartialShape{3}),
                                std::make_shared<Parameter>(f32, PartialShape{8, 1, 8, 4}),
                                std::make_shared<Parameter>(f32, PartialShape{8, 2, 4}),
                                std::make_shared<Parameter>(f32, PartialShape{8, 1, 2, 4})};

    PagedQSAIndexer::Config config;
    config.key_block_size = 4;
    config.block_topk = 3;
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = 2;
    config.indexer_head_dim = 4;
    config.epsilon = 1e-5f;
    config.scale = 0.5f;
    const auto op = std::make_shared<PagedQSAIndexer>(args, config);

    NodeBuilder builder(op, args);
    const auto restored = ov::as_type_ptr<PagedQSAIndexer>(builder.create());
    ASSERT_NE(restored, nullptr);
    const auto& restored_config = restored->get_config();
    EXPECT_EQ(restored_config.key_block_size, config.key_block_size);
    EXPECT_EQ(restored_config.block_topk, config.block_topk);
    EXPECT_EQ(restored_config.num_selection_heads, config.num_selection_heads);
    EXPECT_EQ(restored_config.indexer_rotary_dim, config.indexer_rotary_dim);
    EXPECT_EQ(restored_config.indexer_head_dim, config.indexer_head_dim);
    EXPECT_EQ(restored_config.epsilon, config.epsilon);
    EXPECT_EQ(restored_config.scale, config.scale);
    EXPECT_EQ(restored->get_output_size(), 2);
}
