// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

namespace ov::test {

TEST(attributes, paged_gated_delta_net_default_attrs) {
    NodeBuilder::opset().insert<op::internal::PagedGatedDeltaNet>();
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
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
    auto g_op = as_type_ptr<op::internal::PagedGatedDeltaNet>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_FALSE(op->get_use_qk_l2norm());
    EXPECT_FLOAT_EQ(op->get_q_l2_norm_eps(), 1e-6f);
    EXPECT_FLOAT_EQ(op->get_k_l2_norm_eps(), 1e-6f);

    EXPECT_EQ(g_op->get_use_qk_l2norm(), op->get_use_qk_l2norm());
    EXPECT_FLOAT_EQ(g_op->get_q_l2_norm_eps(), op->get_q_l2_norm_eps());
    EXPECT_FLOAT_EQ(g_op->get_k_l2_norm_eps(), op->get_k_l2_norm_eps());
}

TEST(attributes, paged_gated_delta_net_non_default_attrs) {
    NodeBuilder::opset().insert<op::internal::PagedGatedDeltaNet>();
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
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
                                                                       true,    // use_qk_l2norm
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
    auto g_op = as_type_ptr<op::internal::PagedGatedDeltaNet>(builder.create());

    constexpr auto expected_attr_count = 3;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);

    EXPECT_TRUE(op->get_use_qk_l2norm());
    EXPECT_FLOAT_EQ(op->get_q_l2_norm_eps(), 1e-3f);
    EXPECT_FLOAT_EQ(op->get_k_l2_norm_eps(), 2e-3f);

    EXPECT_EQ(g_op->get_use_qk_l2norm(), op->get_use_qk_l2norm());
    EXPECT_FLOAT_EQ(g_op->get_q_l2_norm_eps(), op->get_q_l2_norm_eps());
    EXPECT_FLOAT_EQ(g_op->get_k_l2_norm_eps(), op->get_k_l2_norm_eps());
}

}  // namespace ov::test
