// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_offsets.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(visitor_without_attribute, embedding_bag_offsets_op) {
    NodeBuilder::opset().insert<ov::op::v15::EmbeddingBagOffsets>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});
    auto reduction = ov::op::v15::EmbeddingBagOffsets::Reduction::SUM;

    auto ebos = make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table,
                                                              indices,
                                                              offsets,
                                                              default_index,
                                                              per_sample_weights,
                                                              reduction);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_op2) {
    NodeBuilder::opset().insert<ov::op::v15::EmbeddingBagOffsets>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto reduction = ov::op::v15::EmbeddingBagOffsets::Reduction::MEAN;

    auto ebos = make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets, default_index, reduction);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_op3) {
    NodeBuilder::opset().insert<ov::op::v15::EmbeddingBagOffsets>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto ebos = make_shared<ov::op::v15::EmbeddingBagOffsets>(emb_table, indices, offsets);
    NodeBuilder builder(ebos, {emb_table, indices, offsets});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
