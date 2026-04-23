// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_gated_delta_net.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::PagedGatedDeltaNet> make_pgdn(const element::Type& data_et,
                                                            const PartialShape& query_ps,
                                                            const PartialShape& key_ps,
                                                            const PartialShape& value_ps,
                                                            const PartialShape& state_ps,
                                                            const PartialShape& gate_ps,
                                                            const PartialShape& beta_ps,
                                                            const PartialShape& subsequence_begins_ps,
                                                            const PartialShape& block_indices_ps,
                                                            const PartialShape& block_indices_begins_ps,
                                                            const PartialShape& past_lens_ps,
                                                            const PartialShape& cache_interval_ps) {
    auto query = std::make_shared<op::v0::Parameter>(data_et, query_ps);
    auto key = std::make_shared<op::v0::Parameter>(data_et, key_ps);
    auto value = std::make_shared<op::v0::Parameter>(data_et, value_ps);
    auto state = std::make_shared<op::v0::Parameter>(data_et, state_ps);
    auto gate = std::make_shared<op::v0::Parameter>(data_et, gate_ps);
    auto beta = std::make_shared<op::v0::Parameter>(data_et, beta_ps);
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, subsequence_begins_ps);
    auto block_indices = std::make_shared<op::v0::Parameter>(element::i32, block_indices_ps);
    auto block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, block_indices_begins_ps);
    auto past_lens = std::make_shared<op::v0::Parameter>(element::i32, past_lens_ps);
    auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, cache_interval_ps);

    return std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{query,
                                                                           key,
                                                                           value,
                                                                           state,
                                                                           gate,
                                                                           beta,
                                                                           subsequence_begins,
                                                                           block_indices,
                                                                           block_indices_begins,
                                                                           past_lens,
                                                                           cache_interval});
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
                              Shape{5},            // block_indices
                              Shape{3},            // block_indices_begins
                              Shape{2},            // past_lens
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
                    testing::HasSubstr("Rank of `query` input should be in [3] list"));
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
                    testing::HasSubstr("Element type of `query` input should be in"));
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
                    testing::HasSubstr("The number of heads in query and key should be the same"));
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
                    testing::HasSubstr("The head size in key and query should be the same"));
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
                    testing::HasSubstr("The number of heads in gate, beta, and value should be the same"));
}

TEST(type_prop, paged_gated_delta_net_state_shape_mismatch) {
    OV_EXPECT_THROW(
        std::ignore = make_pgdn(element::f32,
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
        testing::HasSubstr("The value_head_dim of recurrent_state_table and head size of value should be the same"));
}

TEST(type_prop, paged_gated_delta_net_wrong_input_count) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 4, 8});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::PagedGatedDeltaNet>(OutputVector{p, p, p}),
                    NodeValidationFailure,
                    testing::HasSubstr("PagedGatedDeltaNet expects 11 inputs"));
}
}  // namespace ov::test
