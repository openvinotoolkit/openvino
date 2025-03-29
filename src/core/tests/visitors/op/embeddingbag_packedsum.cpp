// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packedsum.hpp"

#include <gtest/gtest.h>

#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(visitor_without_attribute, embedding_bag_packed_sum_op) {
    NodeBuilder::opset().insert<ov::op::v3::EmbeddingBagPackedSum>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<ov::op::v0::Parameter>(element::f32, Shape{3, 4});

    auto ebps = make_shared<ov::op::v3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
    NodeBuilder builder(ebps, {emb_table, indices, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<ov::op::v3::EmbeddingBagPackedSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_packed_sum_op2) {
    NodeBuilder::opset().insert<ov::op::v3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<ov::op::v0::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<ov::op::v0::Parameter>(element::i64, Shape{3, 4});

    auto ebps = make_shared<ov::op::v3::EmbeddingBagPackedSum>(emb_table, indices);
    NodeBuilder builder(ebps, {emb_table, indices});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<ov::op::v3::EmbeddingBagPackedSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
