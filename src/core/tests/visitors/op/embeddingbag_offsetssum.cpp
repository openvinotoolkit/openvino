// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<op::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});

    auto ebos =
        make_shared<opset3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index, per_sample_weights);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<opset3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op2) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<op::Parameter>(element::i64, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    auto ebos = make_shared<opset3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets, default_index);
    NodeBuilder builder(ebos, {emb_table, indices, offsets, default_index});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<opset3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_offsets_sum_op3) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto offsets = make_shared<op::Parameter>(element::i64, Shape{4});

    auto ebos = make_shared<opset3::EmbeddingBagOffsetsSum>(emb_table, indices, offsets);
    NodeBuilder builder(ebos, {emb_table, indices, offsets});
    EXPECT_NO_THROW(auto g_ebos = ov::as_type_ptr<opset3::EmbeddingBagOffsetsSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
