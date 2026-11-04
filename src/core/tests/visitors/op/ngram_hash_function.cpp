// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ngram_hash_function.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using ov::PartialShape;
using ov::op::v0::Parameter;
using ov::test::NodeBuilder;

TEST(attributes, ngram_hash_function) {
    using ov::op::internal::NGramHashFunction;
    NodeBuilder::opset().insert<NGramHashFunction>();

    const auto ngrams = std::make_shared<Parameter>(ov::element::i64, PartialShape{-1, 3});
    const auto op = std::make_shared<NGramHashFunction>(ngrams,
                                                        3,
                                                        std::vector<int64_t>{2, 1},
                                                        std::vector<int64_t>{23703573157769,
                                                                             20109073645365,
                                                                             static_cast<int64_t>(0x9E3779B97F4A7C15ULL)},
                                                        std::vector<int64_t>{20000003, 20000023, 20000033},
                                                        std::vector<int64_t>{0, 20000003, 40000026});

    NodeBuilder builder(op, {ngrams});
    auto g_op = ov::as_type_ptr<NGramHashFunction>(builder.create());
    ASSERT_NE(g_op, nullptr);

    constexpr auto expected_attr_count = 5;
    EXPECT_EQ(builder.get_value_map_size(), expected_attr_count);
    EXPECT_EQ(g_op->get_ngram_size(), op->get_ngram_size());
    EXPECT_EQ(g_op->get_heads_per_order(), op->get_heads_per_order());
    EXPECT_EQ(g_op->get_multipliers(), op->get_multipliers());
    EXPECT_EQ(g_op->get_head_vocab_sizes(), op->get_head_vocab_sizes());
    EXPECT_EQ(g_op->get_head_offsets(), op->get_head_offsets());
    EXPECT_EQ(g_op->get_output_element_type(0), op->get_output_element_type(0));
    EXPECT_EQ(g_op->get_output_partial_shape(0), op->get_output_partial_shape(0));
}
