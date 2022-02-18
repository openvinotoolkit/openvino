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

TEST(visitor_without_attribute, embedding_segments_sum_op) {
    NodeBuilder::get_ops().register_factory<opset3::EmbeddingSegmentsSum>();
    auto emb_table = make_shared<op::Parameter>(element::f32, Shape{5, 2, 3});

    auto indices = make_shared<op::Parameter>(element::i64, Shape{4});
    auto segment_ids = make_shared<op::Parameter>(element::i64, Shape{4});
    auto num_segments = make_shared<op::Parameter>(element::i64, Shape{});
    auto per_sample_weights = make_shared<op::Parameter>(element::f32, Shape{4});
    auto default_index = make_shared<op::Parameter>(element::i64, Shape{});

    auto ess = make_shared<opset3::EmbeddingSegmentsSum>(emb_table,
                                                         indices,
                                                         segment_ids,
                                                         num_segments,
                                                         default_index,
                                                         per_sample_weights);
    NodeBuilder builder(ess);

    const auto expected_attr_count = 0;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
}
