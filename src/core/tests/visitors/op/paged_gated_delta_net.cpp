// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, paged_gated_delta_net_default_attrs) {
    NodeBuilder::opset().insert<ov::op::internal::PagedGatedDeltaNet>();
    const auto query = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto key = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto value = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16});
    const auto state = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8, 16});
    const auto gate = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto beta = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto subsequence_begins = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_shared<ov::op::internal::PagedGatedDeltaNet>(OutputVector{query,
                                                                                   key,
                                                                                   value,
                                                                                   state,
                                                                                   gate,
                                                                                   beta,
                                                                                   subsequence_begins,
                                                                                   la_block_indices,
                                                                                   la_block_indices_begins,
                                                                                   processed_tokens,
                                                                                   cache_interval});

    NodeBuilder builder(op,
                        {query,
                         key,
                         value,
                         state,
                         gate,
                         beta,
                         subsequence_begins,
                         la_block_indices,
                         la_block_indices_begins,
                         processed_tokens,
                         cache_interval});
    auto g_op = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_op->get_fuse_qk_l2norm(), op->get_fuse_qk_l2norm());
    EXPECT_NEAR(g_op->get_q_l2_norm_eps(), op->get_q_l2_norm_eps(), 1e-10f);
    EXPECT_NEAR(g_op->get_k_l2_norm_eps(), op->get_k_l2_norm_eps(), 1e-10f);
}

TEST(attributes, paged_gated_delta_net_non_default_attrs) {
    NodeBuilder::opset().insert<ov::op::internal::PagedGatedDeltaNet>();
    const auto query = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto key = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto value = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16});
    const auto state = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8, 16});
    const auto gate = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto beta = make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto subsequence_begins = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = make_shared<ov::op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_shared<ov::op::internal::PagedGatedDeltaNet>(OutputVector{query,
                                                                                   key,
                                                                                   value,
                                                                                   state,
                                                                                   gate,
                                                                                   beta,
                                                                                   subsequence_begins,
                                                                                   la_block_indices,
                                                                                   la_block_indices_begins,
                                                                                   processed_tokens,
                                                                                   cache_interval},
                                                                      true,    // fuse_qk_l2norm
                                                                      1e-3f,   // q_l2_norm_eps
                                                                      2e-3f);  // k_l2_norm_eps

    NodeBuilder builder(op,
                        {query,
                         key,
                         value,
                         state,
                         gate,
                         beta,
                         subsequence_begins,
                         la_block_indices,
                         la_block_indices_begins,
                         processed_tokens,
                         cache_interval});
    auto g_op = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_EQ(g_op->get_fuse_qk_l2norm(), op->get_fuse_qk_l2norm());
    EXPECT_NEAR(g_op->get_q_l2_norm_eps(), op->get_q_l2_norm_eps(), 1e-10f);
    EXPECT_NEAR(g_op->get_k_l2_norm_eps(), op->get_k_l2_norm_eps(), 1e-10f);
}
