// Copyright (C) 2018-2023 Intel Corporation
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

TEST(visitor_without_attribute, embedding_bag_packed_sum_op) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{3, 4});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{3, 4});

    auto ebps = make_shared<opset3::EmbeddingBagPackedSum>(emb_table, indices, per_sample_weights);
    NodeBuilder builder(ebps, {emb_table, indices, per_sample_weights});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<opset3::EmbeddingBagPackedSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}

TEST(visitor_without_attribute, embedding_bag_packed_sum_op2) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingBagOffsetsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2});
    auto indices = make_shared<op::Parameter>(element::i64, Shape{3, 4});

    auto ebps = make_shared<opset3::EmbeddingBagPackedSum>(emb_table, indices);
    NodeBuilder builder(ebps, {emb_table, indices});
    EXPECT_NO_THROW(auto g_ebps = ov::as_type_ptr<opset3::EmbeddingBagPackedSum>(builder.create()));

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
