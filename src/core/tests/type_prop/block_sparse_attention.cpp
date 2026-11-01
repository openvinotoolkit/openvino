// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental/block_sparse_attention.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/op/parameter.hpp"

namespace ov::test {

class TypePropBlockSparseAttentionTest : public TypePropOpTest<op::experimental::BlockSparseAttention> {};

TEST_F(TypePropBlockSparseAttentionTest, static_shapes_4_input) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 8, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 2, 2});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 256, 8, 64}));
    EXPECT_EQ(op->get_block_size_q(), 128);
    EXPECT_EQ(op->get_block_size_kv(), 128);
    EXPECT_FALSE(op->get_causal());
    EXPECT_FALSE(op->has_evaluate());
}

TEST_F(TypePropBlockSparseAttentionTest, u8_block_mask_accepted) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 8, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 256, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::u8, PartialShape{1, 1, 2, 2});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 256, 8, 64}));
}

TEST_F(TypePropBlockSparseAttentionTest, gqa_hq_hkv) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 8, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 128, 8, 64}));
}

TEST_F(TypePropBlockSparseAttentionTest, hq_not_divisible_by_hkv_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 5, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("must be evenly divisible by Hkv"));
}

TEST_F(TypePropBlockSparseAttentionTest, ev_may_differ_from_e) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 96});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 128, 2, 96}));
}

// Regression test: key/value batch must NOT silently broadcast against query's batch --
// evaluate()/the reference/GPU kernels index key/value with query's own batch index directly, with
// no broadcast fallback, so a mismatched key/value batch must be rejected outright rather than
// accepted via DimType::broadcast_merge.
TEST_F(TypePropBlockSparseAttentionTest, mismatched_key_batch_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{4, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("Batch dim of key"));
}

TEST_F(TypePropBlockSparseAttentionTest, mismatched_value_batch_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{4, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("Batch dim of value"));
}

// block_mask's Bm is the one input allowed to broadcast (1 or B); output batch is still exactly B.
TEST_F(TypePropBlockSparseAttentionTest, broadcast_block_mask_batch_allowed) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{4, 128, 2, 64}));
}

TEST_F(TypePropBlockSparseAttentionTest, mismatched_block_mask_batch_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{4, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{2, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("block_mask batch dim"));
}

// Lq/Lk need not be evenly divisible by block_size_q/block_size_kv -- the last block may be a
// shorter tail block (see the reference implementation and block-sparse-attention.rst).
TEST_F(TypePropBlockSparseAttentionTest, tail_block_accepted) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    // ceil(200 / 128) == 2.
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 2, 2});

    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 200, 2, 64}));
}

TEST_F(TypePropBlockSparseAttentionTest, wrong_mask_block_count_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 200, 2, 64});
    // Wrong: exact division (200/128 == 1), not ceil (2).
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 2});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("query-block dim"));
}

TEST_F(TypePropBlockSparseAttentionTest, dynamic_5_input_with_scale) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(4));
    const auto scale = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{});

    const auto op = make_op(query, key, value, block_mask, scale, /*block_size_q=*/128, /*block_size_kv=*/128);

    EXPECT_EQ(op->get_input_size(), 5u);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic(4));
    EXPECT_FALSE(op->has_evaluate());
}

TEST_F(TypePropBlockSparseAttentionTest, non_floating_point_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("floating-point"));
}

TEST_F(TypePropBlockSparseAttentionTest, non_positive_block_size_throws) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});

    OV_EXPECT_THROW(std::ignore = make_op(query, key, value, block_mask, /*block_size_q=*/0, /*block_size_kv=*/128),
                    NodeValidationFailure,
                    testing::HasSubstr("block_size_q must be positive"));
}

TEST_F(TypePropBlockSparseAttentionTest, clone_with_new_inputs_4_input) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{1, 128, 2, 64});
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape{1, 1, 1, 1});
    const auto op = make_op(query, key, value, block_mask, /*block_size_q=*/128, /*block_size_kv=*/128);

    const auto cloned = op->clone_with_new_inputs(op->input_values());

    EXPECT_EQ(cloned->get_input_size(), 4u);
    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

TEST_F(TypePropBlockSparseAttentionTest, clone_with_new_inputs_5_input) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, PartialShape::dynamic(4));
    const auto block_mask = std::make_shared<op::v0::Parameter>(element::boolean, PartialShape::dynamic(4));
    const auto scale = std::make_shared<op::v0::Parameter>(element::f16, PartialShape{});
    const auto op = make_op(query, key, value, block_mask, scale, /*block_size_q=*/128, /*block_size_kv=*/128);

    const auto cloned = op->clone_with_new_inputs(op->input_values());

    EXPECT_EQ(cloned->get_input_size(), 5u);
    EXPECT_EQ(cloned->get_output_partial_shape(0), op->get_output_partial_shape(0));
}

}  // namespace ov::test
