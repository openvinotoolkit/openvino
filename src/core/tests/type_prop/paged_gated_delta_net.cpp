// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::PagedGatedDeltaNet> make_pgdn(const element::Type& query_et,
                                                            const element::Type& key_et,
                                                            const element::Type& value_et,
                                                            const element::Type& state_et,
                                                            const element::Type& gate_et,
                                                            const element::Type& beta_et,
                                                            const PartialShape& query_ps,
                                                            const PartialShape& key_ps,
                                                            const PartialShape& value_ps,
                                                            const PartialShape& state_ps,
                                                            const PartialShape& gate_ps,
                                                            const PartialShape& beta_ps,
                                                            const PartialShape& subsequence_begins_ps,
                                                            const PartialShape& la_block_indices_ps,
                                                            const PartialShape& la_block_indices_begins_ps,
                                                            const PartialShape& processed_tokens_ps,
                                                            const PartialShape& cache_interval_ps,
                                                            const bool use_qk_l2norm = false,
                                                            const float q_l2_norm_eps = 1e-6F,
                                                            const float k_l2_norm_eps = 1e-6F) {
    auto query = std::make_shared<op::v0::Parameter>(query_et, query_ps);
    auto key = std::make_shared<op::v0::Parameter>(key_et, key_ps);
    auto value = std::make_shared<op::v0::Parameter>(value_et, value_ps);
    auto state = std::make_shared<op::v0::Parameter>(state_et, state_ps);
    auto gate = std::make_shared<op::v0::Parameter>(gate_et, gate_ps);
    auto beta = std::make_shared<op::v0::Parameter>(beta_et, beta_ps);
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, subsequence_begins_ps);
    auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, la_block_indices_ps);
    auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, la_block_indices_begins_ps);
    auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, processed_tokens_ps);
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, cache_interval_ps);

    return std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
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
                                                              use_qk_l2norm,
                                                              q_l2_norm_eps,
                                                              k_l2_norm_eps);
}

std::shared_ptr<op::internal::PagedGatedDeltaNet> make_pgdn(const element::Type& data_et,
                                                            const PartialShape& query_ps,
                                                            const PartialShape& key_ps,
                                                            const PartialShape& value_ps,
                                                            const PartialShape& state_ps,
                                                            const PartialShape& gate_ps,
                                                            const PartialShape& beta_ps,
                                                            const PartialShape& subsequence_begins_ps,
                                                            const PartialShape& la_block_indices_ps,
                                                            const PartialShape& la_block_indices_begins_ps,
                                                            const PartialShape& processed_tokens_ps,
                                                            const PartialShape& cache_interval_ps,
                                                            const bool use_qk_l2norm = false,
                                                            const float q_l2_norm_eps = 1e-6F,
                                                            const float k_l2_norm_eps = 1e-6F) {
    return make_pgdn(data_et,
                     data_et,
                     data_et,
                     data_et,
                     data_et,
                     data_et,
                     query_ps,
                     key_ps,
                     value_ps,
                     state_ps,
                     gate_ps,
                     beta_ps,
                     subsequence_begins_ps,
                     la_block_indices_ps,
                     la_block_indices_begins_ps,
                     processed_tokens_ps,
                     cache_interval_ps,
                     use_qk_l2norm,
                     q_l2_norm_eps,
                     k_l2_norm_eps);
}

}  // namespace

TEST(type_prop, paged_gated_delta_net_static_f32) {
    const auto op = make_pgdn(element::f32,
                              Shape{10, 4, 8},     // query
                              Shape{10, 4, 8},     // key
                              Shape{10, 4, 16},    // value
                              Shape{5, 4, 8, 16},  // recurrent_state_table
                              Shape{10, 4},        // gate
                              Shape{10, 4},        // beta
                              Shape{3},            // subsequence_begins
                              Shape{5},            // la_block_indices
                              Shape{3},            // la_block_indices_begins
                              Shape{2},            // processed_tokens
                              Shape{2});           // cache_interval

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 4, 16}));
}

TEST(type_prop, paged_gated_delta_net_static_f16) {
    const auto op = make_pgdn(element::f16,
                              Shape{10, 4, 8},
                              Shape{10, 4, 8},
                              Shape{10, 4, 16},
                              Shape{5, 4, 8, 16},
                              Shape{10, 4},
                              Shape{10, 4},
                              Shape{3},
                              Shape{5},
                              Shape{3},
                              Shape{2},
                              Shape{2});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 4, 16}));
}

TEST(type_prop, paged_gated_delta_net_partial_shape_infer) {
    const auto op = make_pgdn(element::bf16,
                              PartialShape{-1, {2, 8}, 64},
                              PartialShape{-1, {2, 8}, 64},
                              PartialShape{-1, {2, 8}, {32, 128}},
                              PartialShape{-1, {2, 8}, 64, {32, 128}},
                              PartialShape{-1, {2, 8}},
                              PartialShape{-1, {2, 8}},
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {2, 8}, {32, 128}}));
}

TEST(type_prop, paged_gated_delta_net_invalid_query_rank) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8, 1},  // rank 4 - invalid
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `query` input must be one of [3]."));
}

TEST(type_prop, paged_gated_delta_net_invalid_type) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::i32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Element type of `query` input must be one of [f32, f16, bf16]."));
}

TEST(type_prop, paged_gated_delta_net_head_num_mismatch_qk) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 6, 8},  // num_heads 6 != 4
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in query and key inputs must be equal"));
}

TEST(type_prop, paged_gated_delta_net_head_size_mismatch_qk) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 32},  // key_head_dim 32 != 8
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 32, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("The head size of query and key inputs must be equal"));
}

TEST(type_prop, paged_gated_delta_net_gate_beta_head_num_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 6},  // gate heads 6 != 4
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("The number of heads in gate, beta, and value inputs must be equal"));
}

TEST(type_prop, paged_gated_delta_net_state_shape_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 32},  // value_head_dim 32 != 16
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr(
                        "The value dimension of recurrent_state_table and the head size of value input must be equal"));
}

TEST(type_prop, paged_gated_delta_net_wrong_input_count) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{p, p, p}),
                    NodeValidationFailure,
                    testing::HasSubstr("PagedGatedDeltaNet expects 11 inputs. Got: 3"));
}

TEST(type_prop, paged_gated_delta_net_dynamic_rank_accepted) {
    // Dynamic rank inputs should be accepted during model import / shape propagation
    const auto op = make_pgdn(element::f32,
                              PartialShape::dynamic(),  // query - dynamic rank
                              PartialShape::dynamic(),  // key - dynamic rank
                              PartialShape::dynamic(),  // value - dynamic rank
                              PartialShape::dynamic(),  // recurrent_state_table - dynamic rank
                              PartialShape::dynamic(),  // gate - dynamic rank
                              PartialShape::dynamic(),  // beta - dynamic rank
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1},
                              PartialShape{-1});

    EXPECT_EQ(op->get_output_size(), 1);
}

TEST(type_prop, paged_gated_delta_net_dynamic_type_accepted) {
    // Dynamic element type inputs should be accepted during model import / shape propagation
    const auto op = make_pgdn(element::dynamic,
                              Shape{10, 4, 8},
                              Shape{10, 4, 8},
                              Shape{10, 4, 16},
                              Shape{5, 4, 8, 16},
                              Shape{10, 4},
                              Shape{10, 4},
                              Shape{3},
                              Shape{5},
                              Shape{3},
                              Shape{2},
                              Shape{2});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
}

TEST(type_prop, paged_gated_delta_net_dynamic_and_concrete_float_types_merge_to_concrete) {
    const auto op = make_pgdn(element::dynamic,
                              element::f32,
                              element::f32,
                              element::f32,
                              element::f32,
                              element::f32,
                              Shape{10, 4, 8},
                              Shape{10, 4, 8},
                              Shape{10, 4, 16},
                              Shape{5, 4, 8, 16},
                              Shape{10, 4},
                              Shape{10, 4},
                              Shape{3},
                              Shape{5},
                              Shape{3},
                              Shape{2},
                              Shape{2});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
}

TEST(type_prop, paged_gated_delta_net_mixed_float_types_rejected) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            element::f16,
                                            element::f32,
                                            element::f32,
                                            element::f32,
                                            element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("have the same floating-point element type"));
}

TEST(type_prop, paged_gated_delta_net_i64_int_inputs_accepted) {
    auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 8, 16});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i64, Shape{5});
    auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});

    const auto op = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
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

TEST(type_prop, paged_gated_delta_net_invalid_subsequence_begins_rank) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{2, 2},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `subsequence_begins` input must be one of [1]."));
}

TEST(type_prop, paged_gated_delta_net_invalid_subsequence_begins_type) {
    auto query = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    auto key = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    auto value = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 16});
    auto state = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 4, 8, 16});
    auto gate = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    auto beta = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4});
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i16, Shape{3});
    auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
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
        testing::HasSubstr("Element type of `subsequence_begins` input must be one of [i32, i64]."));
}

TEST(type_prop, paged_gated_delta_net_negative_q_eps_rejected) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2},
                                            false,
                                            -1.0f,
                                            1e-6f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'q_l2_norm_eps' must be a positive floating-point number"));
}

TEST(type_prop, paged_gated_delta_net_zero_q_eps_rejected) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2},
                                            false,
                                            0.0f,
                                            1e-6f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'q_l2_norm_eps' must be a positive floating-point number"));
}

TEST(type_prop, paged_gated_delta_net_zero_k_eps_rejected) {
    OV_EXPECT_THROW(std::ignore = make_pgdn(element::f32,
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 8},
                                            Shape{10, 4, 16},
                                            Shape{5, 4, 8, 16},
                                            Shape{10, 4},
                                            Shape{10, 4},
                                            Shape{3},
                                            Shape{5},
                                            Shape{3},
                                            Shape{2},
                                            Shape{2},
                                            false,
                                            1e-6f,
                                            0.0f),
                    NodeValidationFailure,
                    testing::HasSubstr("Attribute 'k_l2_norm_eps' must be a positive floating-point number"));
}
}  // namespace ov::test
