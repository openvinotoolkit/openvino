// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {

class TypePropPagedGatedDeltaNetTest : public TypePropOpTest<op::internal::PagedGatedDeltaNet> {};

TEST_F(TypePropPagedGatedDeltaNetTest, f32_static_shapes) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 4, 16}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, f16_static_shapes) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f16, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 4, 16}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, bf16_partial_shapes) {
    const auto query = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}, {32, 128}});
    const auto state = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}, {32, 128}, 64});
    const auto gate = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}});
    const auto beta = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {2, 8}});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {2, 8}, {32, 128}}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, query_incompatible_rank) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8, 1});  // rank 4 - invalid
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[0].rank().compatible(3)"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, query_incompatible_type) {
    const auto query = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::i32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("Float inputs must have f32, f16, or bf16 element type"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, key_incompatible_head_num) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 6, 8});  // num_heads 6 != 4
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in query and key inputs must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, key_incompatible_head_num_partial) {
    // Non-overlapping dimension ranges: query num_heads {2,4}, key num_heads {6,10}
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {2, 4}, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {6, 10}, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {6, 10}, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {6, 10}, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {6, 10}});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {6, 10}});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in query and key inputs must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, key_incompatible_head_size) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 32});  // key_head_dim 32 != 8
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 32});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head size of query and key inputs must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, gate_incompatible_head_num) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 6});  // gate heads 6 != 4
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in gate, beta, and value inputs must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, state_incompatible_value_dim) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state =
        std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 32, 8});  // value_head_dim 32 != 16
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr(
                        "The value dimension of recurrent_state_table and the head size of value input must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, state_incompatible_num_heads) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 6, 16, 8});  // num_heads 6 != 4
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in recurrent_state_table and value inputs must be equal."));
}

TEST_F(TypePropPagedGatedDeltaNetTest, state_incompatible_key_dim) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 16});  // key_head_dim 16 != 8
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(
        std::ignore = make_op(OutputVector{query,
                                           key,
                                           value,
                                           state,
                                           gate,
                                           beta,
                                           subsequence_begins,
                                           la_block_indices,
                                           la_block_indices_begins,
                                           processed_tokens,
                                           cache_interval}),
        NodeValidationFailure,
        testing::HasSubstr("The key dimension of recurrent_state_table and the head size of key input must be equal."));
}

TEST_F(TypePropPagedGatedDeltaNetTest, state_incompatible_value_dim_partial) {
    // Non-overlapping dimension ranges: value value_head_dim {8,16}, state value_head_dim {32,64}
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, {8, 16}});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4, {32, 64}, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr(
                        "The value dimension of recurrent_state_table and the head size of value input must be equal"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, incompatible_inputs_count) {
    const auto p = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{p, p, p}),
                    NodeValidationFailure,
                    testing::HasSubstr("get_input_size() == 11"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, dynamic_rank) {
    // Dynamic rank inputs should be accepted during model import / shape propagation
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape::dynamic());
}

TEST_F(TypePropPagedGatedDeltaNetTest, dynamic_type) {
    // Dynamic element type inputs should be accepted during model import / shape propagation
    const auto query = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{10, 4, 16}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, dynamic_and_concrete_float_type_merged) {
    const auto query = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{10, 4, 16}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, overlapping_dim_ranges_compatible) {
    // Overlapping (not identical) ranges are compatible: query num_heads {2,6}, key num_heads {4,8}
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {2, 6}, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 8}, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 6}, {16, 32}});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 6}, {16, 32}, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 6}});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 6}});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {4, 6}, {16, 32}}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, mixed_float_types_incompatible) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("have the same element type"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, i64_integer_inputs) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i64, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 4, 16}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, subsequence_begins_incompatible_rank) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 2});  // rank 2 - invalid
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[6].rank().compatible(1)"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, subsequence_begins_incompatible_type) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i16, Shape{3});  // i16 - invalid
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("Integer inputs must have i32 or i64 element type"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, q_l2norm_eps_incompatible_negative) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval},
                                          false,
                                          -1.0f,
                                          1e-6f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'q_l2_norm_eps' must be a positive floating-point number"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, q_l2norm_eps_incompatible_zero) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval},
                                          false,
                                          0.0f,
                                          1e-6f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'q_l2_norm_eps' must be a positive floating-point number"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, k_l2norm_eps_incompatible_zero) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 16, 8});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{query,
                                                       key,
                                                       value,
                                                       state,
                                                       gate,
                                                       beta,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval},
                                          false,
                                          1e-6f,
                                          0.0f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'k_l2_norm_eps' must be a positive floating-point number"));
}

TEST_F(TypePropPagedGatedDeltaNetTest, gqa_static_shapes) {
    // GQA: 2 q/k heads, 8 v heads (4 groups)
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 2, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 2, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 8, 32});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 8, 32, 64});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 8});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 8});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 8, 32}));
}

TEST_F(TypePropPagedGatedDeltaNetTest, gqa_partial_shapes) {
    // GQA with partial shapes: q/k heads {1,4}, v heads {4,16}
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {1, 4}, 64});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {1, 4}, 64});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 16}, {32, 128}});
    const auto state = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 16}, {32, 128}, 64});
    const auto gate = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 16}});
    const auto beta = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{-1, {4, 16}});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{query,
                                         key,
                                         value,
                                         state,
                                         gate,
                                         beta,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {4, 16}, {32, 128}}));
}
}  // namespace ov::test
