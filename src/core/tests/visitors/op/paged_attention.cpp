// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::test::NodeBuilder;

TEST(attributes, paged_attention) {
    NodeBuilder::opset().insert<ov::op::PagedAttentionExtension>();
    const auto query = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 16});
    const auto key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 16});
    const auto value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 16});
    const auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 4, 10, 4});
    const auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{10, 4, 10, 4});
    const auto past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{4});
    const auto subsequence_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
    const auto block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{10});
    const auto block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{5});
    const auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{});
    const auto sliding_window = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
    const auto alibi_slopes = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    const auto max_context_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});

    const auto paged_attention = std::make_shared<ov::op::PagedAttentionExtension>(query,
                                                                                   key,
                                                                                   value,
                                                                                   key_cache,
                                                                                   value_cache,
                                                                                   past_lens,
                                                                                   subsequence_begins,
                                                                                   block_indices,
                                                                                   block_indices_begins,
                                                                                   scale,
                                                                                   sliding_window,
                                                                                   alibi_slopes,
                                                                                   max_context_len);
    NodeBuilder builder(paged_attention,
                        {query,
                         key,
                         value,
                         key_cache,
                         value_cache,
                         past_lens,
                         subsequence_begins,
                         block_indices,
                         block_indices_begins,
                         scale,
                         sliding_window,
                         alibi_slopes,
                         max_context_len});
    auto g_paged_attention = ov::as_type_ptr<ov::op::PagedAttentionExtension>(builder.create());

    constexpr auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_paged_attention->get_out_type(0), paged_attention->get_out_type(0));
    EXPECT_EQ(g_paged_attention->get_out_type(1), paged_attention->get_out_type(1));
}
