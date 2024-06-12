// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packed.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(visitor_without_attribute, embedding_bag_packed_op) {
    NodeBuilder::opset().insert<ov::op::v15::EmbeddingBagPacked>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});
    auto reduction = ov::op::v15::EmbeddingBagPacked::Reduction::SUM;

    auto ebps = make_shared<ov::op::v15::EmbeddingBagPacked>(emb_table, indices, per_sample_weights, reduction);
    NodeBuilder builder(ebps, {emb_table, indices, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<ov::op::v15::EmbeddingBagPacked>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_packed_op2) {
    NodeBuilder::opset().insert<ov::op::v15::EmbeddingBagPacked>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto reduction = ov::op::v15::EmbeddingBagPacked::Reduction::MEAN;

    auto ebps = make_shared<ov::op::v15::EmbeddingBagPacked>(emb_table, indices, reduction);
    NodeBuilder builder(ebps, {emb_table, indices});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<ov::op::v15::EmbeddingBagPacked>(builder.create()));

    const auto expected_attr_count = 1;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
