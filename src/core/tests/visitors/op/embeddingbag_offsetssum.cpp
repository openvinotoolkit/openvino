// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op) {
    NodeBuilder::opset().insert<ov::op::v3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{4});

    auto ebos =
        make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index, per_sample_weights);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op2) {
    NodeBuilder::opset().insert<ov::op::v3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<ov::op::v0::Parameter>(element::i64, Shape{});

    auto ebos = make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op3) {
    NodeBuilder::opset().insert<ov::op::v3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<ov::op::v0::Parameter>(element::i64, Shape{4});

    auto ebos = make_shared<ov::op::v3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets);
    NodeBuilder builder(ebos, {emb_table, indices, offsets});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<ov::op::v3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
