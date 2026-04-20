// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, paged_causal_conv1d_roundtrip) {
    using ov::op::internal::PagedCausalConv1D;
    NodeBuilder::opset().insert<PagedCausalConv1D>();

    const auto input_embeds = std::make_shared<Parameter>(ov::element::f32, PartialShape{4, 8});
    const auto conv_state_table = std::make_shared<Parameter>(ov::element::f32, PartialShape{2, 8, 4});
    const auto conv_weight = std::make_shared<Parameter>(ov::element::f32, PartialShape{8, 1, 4});
    const auto conv_bias = std::make_shared<Parameter>(ov::element::f32, PartialShape{8});
    const auto subsequence_begins = std::make_shared<Parameter>(ov::element::i32, PartialShape{2});
    const auto la_block_indices = std::make_shared<Parameter>(ov::element::i32, PartialShape{2});
    const auto la_block_indices_begins = std::make_shared<Parameter>(ov::element::i32, PartialShape{2});
    const auto processed_tokens = std::make_shared<Parameter>(ov::element::i32, PartialShape{1});
    const auto cache_interval = std::make_shared<Parameter>(ov::element::i32, PartialShape{1});

    const auto op = std::make_shared<PagedCausalConv1D>(input_embeds,
                                                        conv_state_table,
                                                        conv_weight,
                                                        conv_bias,
                                                        subsequence_begins,
                                                        la_block_indices,
                                                        la_block_indices_begins,
                                                        processed_tokens,
                                                        cache_interval);

    NodeBuilder builder(op,
                        {input_embeds,
                         conv_state_table,
                         conv_weight,
                         conv_bias,
                         subsequence_begins,
                         la_block_indices,
                         la_block_indices_begins,
                         processed_tokens,
                         cache_interval});
    auto g_op = ov::as_type_ptr<PagedCausalConv1D>(builder.create());

    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
