// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/block_sparse_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"

using namespace ov;
using namespace testing;

namespace {
using ov::op::v0::Parameter;
using ov::op::v17::BlockSparseAttention;

std::shared_ptr<Parameter> make_param(const element::Type& type, const PartialShape& shape) {
    return std::make_shared<Parameter>(type, shape);
}
}  // namespace

TEST(type_prop, block_sparse_attention_static_minimal) {
    // batch=2, heads=4, L=8, E=16, S=32, Ev=16, block_size=4 -> num_q_blocks=2, k_blocks=3
    const auto query = make_param(element::f32, PartialShape{2, 4, 8, 16});
    const auto key = make_param(element::f32, PartialShape{2, 4, 32, 16});
    const auto value = make_param(element::f32, PartialShape{2, 4, 32, 16});
    const auto block_indices = make_param(element::i32, PartialShape{2, 4, 2, 3});

    const auto op = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{2, 4, 8, 16}));
    EXPECT_EQ(op->get_block_size(), 4);
    EXPECT_FALSE(op->get_causal());
}

TEST(type_prop, block_sparse_attention_with_mask) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 24});
    const auto block_indices = make_param(element::i64, PartialShape{1, 2, 2, 2});
    const auto block_indices_mask = make_param(element::boolean, PartialShape{1, 2, 2, 2});

    const auto op =
        std::make_shared<BlockSparseAttention>(query, key, value, block_indices, block_indices_mask, 4, true);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 8, 24}));
    EXPECT_TRUE(op->get_causal());
}

TEST(type_prop, block_sparse_attention_with_u8_mask) {
    // u8 must be accepted as an alternative to boolean for `block_indices_mask`: plugins such as
    // CPU normalize `boolean` graph tensors to `u8` storage (via ov::pass::ConvertPrecision)
    // before any op executes, so a graph built with a genuine boolean Parameter ends up feeding
    // this op a u8 tensor once compiled -- this op must tolerate that, exactly like it must
    // tolerate genuine boolean.
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 24});
    const auto block_indices = make_param(element::i64, PartialShape{1, 2, 2, 2});
    const auto block_indices_mask = make_param(element::u8, PartialShape{1, 2, 2, 2});

    const auto op =
        std::make_shared<BlockSparseAttention>(query, key, value, block_indices, block_indices_mask, 4, true);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 8, 24}));
    EXPECT_TRUE(op->get_causal());
}

TEST(type_prop, block_sparse_attention_with_mask_and_scale) {
    const auto query = make_param(element::f16, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f16, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f16, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});
    const auto block_indices_mask = make_param(element::boolean, PartialShape{1, 2, 2, 2});
    const auto scale = make_param(element::f16, PartialShape{});

    const auto op = std::make_shared<BlockSparseAttention>(query,
                                                           key,
                                                           value,
                                                           block_indices,
                                                           block_indices_mask,
                                                           scale,
                                                           4,
                                                           false);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 2, 8, 16}));
}

TEST(type_prop, block_sparse_attention_broadcast_single_kv_head) {
    // 8 query heads, a single shared kv head and a single shared block-selection head,
    // broadcast to every query head (the level of head broadcasting this op supports;
    // arbitrary GQA grouping, e.g. Hk=2 with H=8, is intentionally not supported -- such
    // models are expected to expand key/value with Broadcast/Tile before this op, exactly
    // as they would before ScaledDotProductAttention).
    const auto query = make_param(element::f32, PartialShape{1, 8, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 1, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 1, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 1, 2, 2});

    const auto op = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 8, 8, 16}));
}

TEST(type_prop, block_sparse_attention_dynamic_batch_and_length) {
    const auto query = make_param(element::f32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic(), 16});
    const auto key = make_param(element::f32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic(), 16});
    const auto value = make_param(element::f32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic(), 16});
    const auto block_indices =
        make_param(element::i32, PartialShape{Dimension::dynamic(), 4, Dimension::dynamic(), 3});

    const auto op = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false);

    EXPECT_EQ(op->get_output_partial_shape(0),
              (PartialShape{Dimension::dynamic(), 4, Dimension::dynamic(), 16}));
}

TEST(type_prop, block_sparse_attention_dynamic_rank_inputs) {
    const auto query = make_param(element::f32, PartialShape::dynamic());
    const auto key = make_param(element::f32, PartialShape::dynamic());
    const auto value = make_param(element::f32, PartialShape::dynamic());
    const auto block_indices = make_param(element::i32, PartialShape::dynamic());

    const auto op = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false);

    EXPECT_EQ(op->get_output_partial_shape(0).rank(), (Rank(4)));
}

TEST(type_prop, block_sparse_attention_incompatible_data_types) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f16, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("element types"));
}

TEST(type_prop, block_sparse_attention_non_integer_block_indices) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::f32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("integer"));
}

TEST(type_prop, block_sparse_attention_non_boolean_mask) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});
    const auto block_indices_mask = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(
        std::ignore =
            std::make_shared<BlockSparseAttention>(query, key, value, block_indices, block_indices_mask, 4, false),
        NodeValidationFailure,
        HasSubstr("boolean"));
}

TEST(type_prop, block_sparse_attention_rank_mismatch) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("rank 4"));
}

TEST(type_prop, block_sparse_attention_head_size_mismatch) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 32});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("head size"));
}

TEST(type_prop, block_sparse_attention_key_value_length_mismatch) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 20, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("number of tokens"));
}

TEST(type_prop, block_sparse_attention_query_length_not_multiple_of_block_size) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 9, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 3, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("multiple of 'block_size'"));
}

TEST(type_prop, block_sparse_attention_key_length_not_multiple_of_block_size) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 18, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 18, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("multiple of 'block_size'"));
}

TEST(type_prop, block_sparse_attention_num_query_blocks_mismatch) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    // query has 2 blocks (8/4), but block_indices says 3
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 3, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 4, false),
                    NodeValidationFailure,
                    HasSubstr("num_query_blocks"));
}

TEST(type_prop, block_sparse_attention_invalid_block_size) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto value = make_param(element::f32, PartialShape{1, 2, 16, 16});
    const auto block_indices = make_param(element::i32, PartialShape{1, 2, 2, 2});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(query, key, value, block_indices, 0, false),
                    NodeValidationFailure,
                    HasSubstr("block_size"));
}

TEST(type_prop, block_sparse_attention_wrong_input_count) {
    const auto query = make_param(element::f32, PartialShape{1, 2, 8, 16});
    const auto key = make_param(element::f32, PartialShape{1, 2, 16, 16});

    OV_EXPECT_THROW(std::ignore = std::make_shared<BlockSparseAttention>(OutputVector{query, key}, 4, false),
                    NodeValidationFailure,
                    HasSubstr("4 to 6 inputs"));
}
