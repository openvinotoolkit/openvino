// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention_indexer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"

namespace ov::test {
using op::v0::Parameter;
using testing::HasSubstr;

class TypePropSparseAttentionIndexerTest : public TypePropOpTest<op::v17::SparseAttentionIndexer> {
protected:
    static std::shared_ptr<Parameter> param(element::Type et, const PartialShape& shape) {
        return std::make_shared<Parameter>(et, shape);
    }
};

TEST_F(TypePropSparseAttentionIndexerTest, default_ctor) {
    const auto op = make_op();
    op->set_key_block_size(4);
    op->set_token_budget(2048);
    op->set_arguments(OutputVector{param(element::f32, PartialShape{1, 4, 7, 128}),
                                   param(element::f32, PartialShape{1, 2500, 128}),
                                   param(element::i64, PartialShape{1})});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 3);
    EXPECT_EQ(op->get_output_size(), 2);
    EXPECT_EQ(op->get_output_element_type(0), element::i32);
    EXPECT_EQ(op->get_output_element_type(1), element::i32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, 7, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 1, 7}));
}

TEST_F(TypePropSparseAttentionIndexerTest, static_shapes) {
    const auto op = make_op(param(element::f32, PartialShape{2, 4, 7, 128}),
                            param(element::f32, PartialShape{2, 2500, 128}),
                            param(element::i32, PartialShape{2}),
                            4,
                            2048);

    EXPECT_EQ(op->get_key_block_size(), 4);
    EXPECT_EQ(op->get_token_budget(), 2048);
    EXPECT_EQ(op->get_num_selection_heads(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 7, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, 7}));
}

TEST_F(TypePropSparseAttentionIndexerTest, batch_mismatch_with_dynamic_rank_index_query) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape::dynamic()),
                                          param(element::f32, PartialShape{2, 2500, 128}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{3, 7, 4}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of index_query, block_index_keys and index_weights "
                              "must be compatible."));
}

TEST_F(TypePropSparseAttentionIndexerTest, key_length_size_checked_against_block_index_keys) {
    // The batch is known from block_index_keys even though index_query has a dynamic rank.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape::dynamic()),
                                          param(element::f32, PartialShape{2, 16, 128}),
                                          param(element::i32, PartialShape{3}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The size of key_length must be 1 or the batch dimension of index_query."));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_key_length_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 4, 7, 128}),
                                          param(element::f32, PartialShape{2, 2500, 128}),
                                          param(element::i32, PartialShape{3}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The size of key_length must be 1 or the batch dimension of index_query."));
}

TEST_F(TypePropSparseAttentionIndexerTest, single_element_key_length_applies_to_all_sequences) {
    const auto op = make_op(param(element::f32, PartialShape{2, 4, 7, 128}),
                            param(element::f32, PartialShape{2, 2500, 128}),
                            param(element::i32, PartialShape{1}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 7, 512}));
}

TEST_F(TypePropSparseAttentionIndexerTest, with_index_weights) {
    const auto op = make_op(param(element::f16, PartialShape{1, 64, 1, 128}),
                            param(element::f16, PartialShape{1, 10001, 128}),
                            param(element::i32, PartialShape{1}),
                            param(element::f16, PartialShape{1, 1, 64}),
                            1,
                            2048);

    EXPECT_EQ(op->get_input_size(), 4);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, 1, 2048}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 1, 1}));
}

TEST_F(TypePropSparseAttentionIndexerTest, multiple_selection_heads) {
    const auto op = make_op(param(element::f32, PartialShape{1, 8, 7, 128}),
                            param(element::f32, PartialShape{1, 100, 128}),
                            param(element::i32, PartialShape{1}),
                            4,
                            64,
                            2);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 7, 16}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 7}));
}

TEST_F(TypePropSparseAttentionIndexerTest, dynamic_dimensions_are_merged) {
    const auto op = make_op(param(element::f32, PartialShape{-1, 4, -1, 128}),
                            param(element::f32, PartialShape{2, -1, -1}),
                            param(element::i32, PartialShape{-1}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, -1, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, -1}));
}

TEST_F(TypePropSparseAttentionIndexerTest, interval_dimensions) {
    const auto op = make_op(param(element::f32, PartialShape{Dimension(1, 4), 4, Dimension(2, 8), 128}),
                            param(element::f32, PartialShape{Dimension(2, 6), -1, 128}),
                            param(element::i32, PartialShape{-1}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension(2, 4), 1, Dimension(2, 8), 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension(2, 4), 1, Dimension(2, 8)}));
}

TEST_F(TypePropSparseAttentionIndexerTest, dynamic_rank_index_query) {
    // The batch is still known, from block_index_keys, which does not broadcast.
    const auto op = make_op(param(element::f32, PartialShape::dynamic()),
                            param(element::f32, PartialShape{2, 2500, 128}),
                            param(element::i32, PartialShape{2}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, -1, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, -1}));
}

TEST_F(TypePropSparseAttentionIndexerTest, size_one_batch_resolves_dynamic_batch) {
    // Batch dimension is defined as 1 based on the second input - no broadcast.
    const auto op = make_op(param(element::f32, PartialShape{-1, 4, 7, 128}),
                            param(element::f32, PartialShape{1, 2500, 128}),
                            param(element::i32, PartialShape{1}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 1, 7, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 1, 7}));
}

TEST_F(TypePropSparseAttentionIndexerTest, size_one_batch_against_static_batch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{4, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{4}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of index_query and block_index_keys must be the same."));
}

TEST_F(TypePropSparseAttentionIndexerTest, all_dynamic_batch) {
    const auto op = make_op(param(element::f32, PartialShape{-1, 4, -1, 128}),
                            param(element::f32, PartialShape{-1, -1, 128}),
                            param(element::i32, PartialShape{-1}),
                            param(element::f32, PartialShape{-1, -1, 4}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, 1, -1, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1, 1, -1}));
}

TEST_F(TypePropSparseAttentionIndexerTest, interval_dimensions_with_index_weights) {
    const auto op = make_op(param(element::f32, PartialShape{Dimension(1, 4), 4, Dimension(2, 8), 128}),
                            param(element::f32, PartialShape{Dimension(2, 6), -1, 128}),
                            param(element::i32, PartialShape{-1}),
                            param(element::f32, PartialShape{Dimension(3, 10), Dimension(4, 16), 4}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension(3, 4), 1, Dimension(4, 8), 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{Dimension(3, 4), 1, Dimension(4, 8)}));
}

TEST_F(TypePropSparseAttentionIndexerTest, dynamic_rank_index_query_with_index_weights) {
    // index_weights is the only source of the query token count once index_query has a dynamic rank.
    const auto op = make_op(param(element::f32, PartialShape::dynamic()),
                            param(element::f32, PartialShape{2, 2500, 128}),
                            param(element::i32, PartialShape{2}),
                            param(element::f32, PartialShape{2, 7, 4}),
                            4,
                            2048);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 1, 7, 512}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{2, 1, 7}));
}

TEST_F(TypePropSparseAttentionIndexerTest, index_heads_taken_from_index_weights) {
    // index_weights is the only source of the index head count once index_query has a dynamic rank.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape::dynamic()),
                                          param(element::f32, PartialShape{2, 2500, 128}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{2, 7, 5}),
                                          4,
                                          2048,
                                          2),
                    NodeValidationFailure,
                    HasSubstr("The number of index query heads must be divisible by num_selection_heads."));
}

TEST_F(TypePropSparseAttentionIndexerTest, symbols_propagation) {
    auto index_query_shape = PartialShape{-1, 4, -1, 128};
    auto index_query_symbols = set_shape_symbols(index_query_shape);

    const auto op = make_op(param(element::f32, index_query_shape),
                            param(element::f32, PartialShape{-1, -1, 128}),
                            param(element::i32, PartialShape{-1}),
                            4,
                            2048);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(1)),
                testing::ElementsAre(index_query_symbols[0], nullptr, index_query_symbols[2]));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_index_query_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The shape of index_query must be 4D [N, Hi, L, Di]."));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_block_index_keys_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The shape of block_index_keys must be 3D [N, B, Di]."));
}

TEST_F(TypePropSparseAttentionIndexerTest, head_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 64}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The head size of index_query and block_index_keys must be the same."));
}

TEST_F(TypePropSparseAttentionIndexerTest, index_weights_heads_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          param(element::f32, PartialShape{1, 7, 8}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The number of index query heads of index_query and index_weights must be the same."));
}

TEST_F(TypePropSparseAttentionIndexerTest, index_query_heads_not_divisible_by_selection_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 5, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048,
                                          2),
                    NodeValidationFailure,
                    HasSubstr("The number of index query heads must be divisible by num_selection_heads."));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_data_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::i32, PartialShape{1, 4, 7, 128}),
                                          param(element::i32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("must be a floating point type"));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_key_length_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::f32, PartialShape{1}),
                                          4,
                                          2048),
                    NodeValidationFailure,
                    HasSubstr("The element type of key_length must be i32 or i64."));
}

TEST_F(TypePropSparseAttentionIndexerTest, token_budget_not_divisible_by_key_block_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2046),
                    NodeValidationFailure,
                    HasSubstr("The attribute token_budget must be divisible by key_block_size."));
}

TEST_F(TypePropSparseAttentionIndexerTest, incorrect_num_selection_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{1, 4, 7, 128}),
                                          param(element::f32, PartialShape{1, 2500, 128}),
                                          param(element::i32, PartialShape{1}),
                                          4,
                                          2048,
                                          0),
                    NodeValidationFailure,
                    HasSubstr("The attribute num_selection_heads must be a positive integer."));
}
}  // namespace ov::test
