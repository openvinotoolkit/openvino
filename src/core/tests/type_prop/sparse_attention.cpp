// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/sparse_attention.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/constant.hpp"

namespace ov::test {
using op::v0::Constant, op::v0::Parameter;
using testing::HasSubstr;

class TypePropSparseAttentionTest : public TypePropOpTest<op::v17::SparseAttention> {
protected:
    static std::shared_ptr<Parameter> param(element::Type et, const PartialShape& shape) {
        return std::make_shared<Parameter>(et, shape);
    }
};

TEST_F(TypePropSparseAttentionTest, default_ctor) {
    const auto op = make_op();
    op->set_key_block_size(4);
    op->set_arguments(OutputVector{param(element::f32, PartialShape{2, 24, 5, 64}),
                                   param(element::f32, PartialShape{2, 2, 12, 64}),
                                   param(element::f32, PartialShape{2, 2, 12, 32}),
                                   param(element::i32, PartialShape{2, 1, 5, 4}),
                                   param(element::i32, PartialShape{2, 1, 5})});
    op->validate_and_infer_types();

    EXPECT_EQ(op->get_input_size(), 5);
    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, 5, 32}));
}

TEST_F(TypePropSparseAttentionTest, static_shapes) {
    const auto op = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                            param(element::f32, PartialShape{2, 2, 12, 64}),
                            param(element::f32, PartialShape{2, 2, 12, 32}),
                            param(element::i32, PartialShape{2, 1, 5, 4}),
                            param(element::i32, PartialShape{2, 1, 5}),
                            4);

    EXPECT_EQ(op->get_key_block_size(), 4);
    EXPECT_EQ(op->get_query_block_size(), 1);
    EXPECT_TRUE(op->get_causal());
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, 5, 32}));
}

TEST_F(TypePropSparseAttentionTest, with_key_length_and_scale) {
    const auto op = make_op(param(element::f16, PartialShape{2, 24, 1, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 32}),
                            param(element::i64, PartialShape{2, 1, 1, 4}),
                            param(element::i64, PartialShape{2, 1, 1}),
                            param(element::i64, PartialShape{2}),
                            param(element::f16, PartialShape{}),
                            4);

    EXPECT_EQ(op->get_input_size(), 7);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, 1, 32}));
}

TEST_F(TypePropSparseAttentionTest, with_sinks) {
    const auto op = make_op(param(element::f16, PartialShape{2, 24, 1, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 32}),
                            param(element::i64, PartialShape{2, 1, 1, 4}),
                            param(element::i64, PartialShape{2, 1, 1}),
                            param(element::i64, PartialShape{2}),
                            param(element::f16, PartialShape{}),
                            param(element::f32, PartialShape{24}),
                            4);

    EXPECT_EQ(op->get_input_size(), 8);
    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, 1, 32}));
}

TEST_F(TypePropSparseAttentionTest, sinks_count_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{}),
                                          param(element::f32, PartialShape{12}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of sinks must be equal to the number of query heads."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_sinks_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{}),
                                          param(element::f32, PartialShape{1, 24}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of sinks must be 1D [H]."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_sinks_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          param(element::i32, PartialShape{2}),
                                          param(element::f32, PartialShape{}),
                                          param(element::i32, PartialShape{24}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element type of sinks must be a floating point type."));
}

TEST_F(TypePropSparseAttentionTest, query_block_size_groups_query_tokens) {
    const auto op = make_op(param(element::f32, PartialShape{1, 12, 8448, 128}),
                            param(element::f32, PartialShape{1, 12, 33792, 128}),
                            param(element::f32, PartialShape{1, 12, 33792, 128}),
                            param(element::i32, PartialShape{1, 12, 66, 264}),
                            param(element::i32, PartialShape{1, 12, 66}),
                            128,
                            128,
                            false);

    EXPECT_EQ(op->get_query_block_size(), 128);
    EXPECT_FALSE(op->get_causal());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 12, 8448, 128}));
}

TEST_F(TypePropSparseAttentionTest, dynamic_dimensions_are_merged) {
    const auto op = make_op(param(element::f32, PartialShape{-1, 24, -1, 64}),
                            param(element::f32, PartialShape{2, 2, -1, -1}),
                            param(element::f32, PartialShape{-1, -1, 12, 32}),
                            param(element::i32, PartialShape{-1, 1, -1, 4}),
                            param(element::i32, PartialShape{-1, -1, -1}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, -1, 32}));
}

TEST_F(TypePropSparseAttentionTest, interval_dimensions) {
    const auto op = make_op(param(element::f32, PartialShape{Dimension(1, 4), 24, Dimension(2, 8), 64}),
                            param(element::f32, PartialShape{Dimension(2, 6), 2, 12, 64}),
                            param(element::f32, PartialShape{-1, 2, 12, Dimension(16, 32)}),
                            param(element::i32, PartialShape{-1, 1, -1, 4}),
                            param(element::i32, PartialShape{-1, 1, -1}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{Dimension(2, 4), 24, Dimension(2, 8), Dimension(16, 32)}));
}

TEST_F(TypePropSparseAttentionTest, dynamic_rank_query) {
    const auto op = make_op(param(element::f32, PartialShape::dynamic()),
                            param(element::f32, PartialShape{2, 2, 12, 64}),
                            param(element::f32, PartialShape{2, 2, 12, 32}),
                            param(element::i32, PartialShape{2, 1, 5, 4}),
                            param(element::i32, PartialShape{2, 1, 5}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropSparseAttentionTest, dynamic_element_type) {
    const auto op = make_op(param(element::dynamic, PartialShape{2, 24, 5, 64}),
                            param(element::f32, PartialShape{2, 2, 12, 64}),
                            param(element::f32, PartialShape{2, 2, 12, 32}),
                            param(element::i32, PartialShape{2, 1, 5, 4}),
                            param(element::i32, PartialShape{2, 1, 5}),
                            4);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST_F(TypePropSparseAttentionTest, integer_scale_with_dynamic_data_types) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::dynamic, PartialShape{2, 24, 1, 64}),
                                          param(element::dynamic, PartialShape{2, 2, 12, 64}),
                                          param(element::dynamic, PartialShape{2, 2, 12, 32}),
                                          param(element::i64, PartialShape{2, 1, 1, 4}),
                                          param(element::i64, PartialShape{2, 1, 1}),
                                          param(element::i64, PartialShape{2}),
                                          param(element::i32, PartialShape{}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element type of scale must be a floating point type"));
}

TEST_F(TypePropSparseAttentionTest, largest_query_block_size_needs_one_query_row) {
    // ceil(L / query_block_size) must not overflow for an extreme block size.
    const auto op = make_op(param(element::f32, PartialShape{1, 24, 8, 64}),
                            param(element::f32, PartialShape{1, 2, 8, 64}),
                            param(element::f32, PartialShape{1, 2, 8, 32}),
                            param(element::i32, PartialShape{1, 1, 1, 4}),
                            param(element::i32, PartialShape{1, 1, 1}),
                            4,
                            std::numeric_limits<int64_t>::max());

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 24, 8, 32}));
}

TEST_F(TypePropSparseAttentionTest, batch_mismatch_with_dynamic_rank_query) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape::dynamic()),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{3, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of all the inputs must be the same."));
}

TEST_F(TypePropSparseAttentionTest, key_length_value_exceeding_key_positions) {
    const auto key_length = Constant::create(element::i64, Shape{2}, {12, 13});
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f16, PartialShape{2, 24, 1, 64}),
                                          param(element::f16, PartialShape{2, 2, 12, 64}),
                                          param(element::f16, PartialShape{2, 2, 12, 32}),
                                          param(element::i64, PartialShape{2, 1, 1, 4}),
                                          param(element::i64, PartialShape{2, 1, 1}),
                                          key_length,
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The key_length values must be in [0, S], S = 12"));
}

TEST_F(TypePropSparseAttentionTest, incorrect_key_length_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f16, PartialShape{2, 24, 1, 64}),
                                          param(element::f16, PartialShape{2, 2, 12, 64}),
                                          param(element::f16, PartialShape{2, 2, 12, 32}),
                                          param(element::i64, PartialShape{2, 1, 1, 4}),
                                          param(element::i64, PartialShape{2, 1, 1}),
                                          param(element::i64, PartialShape{3}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The size of key_length must be 1 or the batch dimension of the other inputs."));
}

TEST_F(TypePropSparseAttentionTest, single_element_key_length_applies_to_all_sequences) {
    const auto op = make_op(param(element::f16, PartialShape{2, 24, 1, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 64}),
                            param(element::f16, PartialShape{2, 2, 12, 32}),
                            param(element::i64, PartialShape{2, 1, 1, 4}),
                            param(element::i64, PartialShape{2, 1, 1}),
                            param(element::i64, PartialShape{1}),
                            4);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 24, 1, 32}));
}

TEST_F(TypePropSparseAttentionTest, zero_selection_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 0, 5, 4}),
                                          param(element::i32, PartialShape{2, 0, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of selection heads must be a positive integer."));
}

TEST_F(TypePropSparseAttentionTest, zero_key_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 0, 12, 64}),
                                          param(element::f32, PartialShape{2, 0, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of key and value heads must be a positive integer."));
}

TEST_F(TypePropSparseAttentionTest, selection_heads_taken_from_block_count) {
    // Hs is dynamic in block_indices, so the divisibility is checked against the merged dimension.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, -1, 5, 4}),
                                          param(element::i32, PartialShape{2, 5, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of query heads must be divisible by the number of selection heads."));
}

TEST_F(TypePropSparseAttentionTest, selection_heads_taken_from_block_count_with_dynamic_rank_block_indices) {
    // block_count is the only source of the selection head count.
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape::dynamic()),
                                          param(element::i32, PartialShape{2, 5, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of query heads must be divisible by the number of selection heads."));
}

TEST_F(TypePropSparseAttentionTest, symbols_propagation) {
    auto query_shape = PartialShape{-1, -1, -1, 64};
    auto value_shape = PartialShape{-1, 2, 12, -1};
    auto query_symbols = set_shape_symbols(query_shape);
    auto value_symbols = set_shape_symbols(value_shape);

    const auto op = make_op(param(element::f32, query_shape),
                            param(element::f32, PartialShape{-1, 2, 12, 64}),
                            param(element::f32, value_shape),
                            param(element::i32, PartialShape{-1, 1, -1, 4}),
                            param(element::i32, PartialShape{-1, 1, -1}),
                            4);

    EXPECT_THAT(get_shape_symbols(op->get_output_partial_shape(0)),
                testing::ElementsAre(query_symbols[0], query_symbols[1], query_symbols[2], value_symbols[3]));
}

TEST_F(TypePropSparseAttentionTest, incorrect_query_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of query must be 4D [N, H, L, E]."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_block_count_rank) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The shape of block_count must be 3D [N, Hs, Lq]."));
}

TEST_F(TypePropSparseAttentionTest, batch_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{3, 2, 12, 64}),
                                          param(element::f32, PartialShape{3, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The batch dimension of all the inputs must be the same."));
}

TEST_F(TypePropSparseAttentionTest, head_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The head size of query and key must be the same."));
}

TEST_F(TypePropSparseAttentionTest, query_heads_not_divisible_by_key_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 25, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of query heads must be divisible by the number of key heads."));
}

TEST_F(TypePropSparseAttentionTest, query_heads_not_divisible_by_selection_heads) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 5, 5, 4}),
                                          param(element::i32, PartialShape{2, 5, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of query heads must be divisible by the number of selection heads."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_number_of_query_rows) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 8, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 8, 4}),
                                          param(element::i32, PartialShape{2, 1, 8}),
                                          4,
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The number of query rows must be ceil(L / query_block_size)"));
}

TEST_F(TypePropSparseAttentionTest, more_queries_than_keys_in_causal_mode) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 16, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 16, 4}),
                                          param(element::i32, PartialShape{2, 1, 16}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("In the causal mode the number of query tokens must not exceed"));
}

TEST_F(TypePropSparseAttentionTest, incorrect_data_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::i32, PartialShape{2, 24, 5, 64}),
                                          param(element::i32, PartialShape{2, 2, 12, 64}),
                                          param(element::i32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("must be a floating point type"));
}

TEST_F(TypePropSparseAttentionTest, mismatched_data_element_types) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f16, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element types of query, key and value must be the same."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_block_indices_element_type) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::f32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          4),
                    NodeValidationFailure,
                    HasSubstr("The element type of block_indices must be i32 or i64."));
}

TEST_F(TypePropSparseAttentionTest, incorrect_key_block_size) {
    OV_EXPECT_THROW(std::ignore = make_op(param(element::f32, PartialShape{2, 24, 5, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 64}),
                                          param(element::f32, PartialShape{2, 2, 12, 32}),
                                          param(element::i32, PartialShape{2, 1, 5, 4}),
                                          param(element::i32, PartialShape{2, 1, 5}),
                                          0),
                    NodeValidationFailure,
                    HasSubstr("The attribute key_block_size must be a positive integer."));
}
}  // namespace ov::test
