// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {
namespace {

std::shared_ptr<op::internal::PagedCausalConv1D> make_pcc(const element::Type& data_et,
                                                          const PartialShape& input_embeds_ps,
                                                          const PartialShape& conv_state_table_ps,
                                                          const PartialShape& conv_weight_ps,
                                                          const PartialShape& conv_bias_ps,
                                                          const PartialShape& subsequence_begins_ps,
                                                          const PartialShape& la_block_indices_ps,
                                                          const PartialShape& la_block_indices_begins_ps,
                                                          const PartialShape& processed_tokens_ps,
                                                          const PartialShape& cache_interval_ps,
                                                          const element::Type& index_et = element::i32) {
    auto input_embeds = std::make_shared<op::v0::Parameter>(data_et, input_embeds_ps);
    auto conv_state_table = std::make_shared<op::v0::Parameter>(data_et, conv_state_table_ps);
    auto conv_weight = std::make_shared<op::v0::Parameter>(data_et, conv_weight_ps);
    auto conv_bias = std::make_shared<op::v0::Parameter>(data_et, conv_bias_ps);
    auto subsequence_begins = std::make_shared<op::v0::Parameter>(index_et, subsequence_begins_ps);
    auto la_block_indices = std::make_shared<op::v0::Parameter>(index_et, la_block_indices_ps);
    auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(index_et, la_block_indices_begins_ps);
    auto processed_tokens = std::make_shared<op::v0::Parameter>(index_et, processed_tokens_ps);
    auto cache_interval = std::make_shared<op::v0::Parameter>(index_et, cache_interval_ps);

    return std::make_shared<op::internal::PagedCausalConv1D>(OutputVector{input_embeds,
                                                                          conv_state_table,
                                                                          conv_weight,
                                                                          conv_bias,
                                                                          subsequence_begins,
                                                                          la_block_indices,
                                                                          la_block_indices_begins,
                                                                          processed_tokens,
                                                                          cache_interval});
}

}  // namespace

TEST(type_prop, paged_causal_conv1d_static_f32) {
    const auto op = make_pcc(element::f32,
                             Shape{10, 256},      // input_embeds
                             Shape{5, 256, 4},    // conv_state_table
                             Shape{256, 256, 4},  // conv_weight
                             Shape{256},          // conv_bias
                             Shape{3},            // subsequence_begins
                             Shape{5},            // la_block_indices
                             Shape{3},            // la_block_indices_begins
                             Shape{2},            // processed_tokens
                             Shape{2});           // cache_interval

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST(type_prop, paged_causal_conv1d_static_f16) {
    const auto op = make_pcc(element::f16,
                             Shape{10, 256},
                             Shape{5, 256, 4},
                             Shape{256, 256, 4},
                             Shape{256},
                             Shape{3},
                             Shape{5},
                             Shape{3},
                             Shape{2},
                             Shape{2});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST(type_prop, paged_causal_conv1d_partial_shape_infer) {
    const auto op = make_pcc(element::bf16,
                             PartialShape{-1, {128, 512}},
                             PartialShape{-1, {128, 512}, -1},
                             PartialShape{-1, -1, -1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {128, 512}}));
}

TEST(type_prop, paged_causal_conv1d_invalid_input_embeds_rank) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256, 1},  // rank 3 - invalid
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `input_embeds` input must be one of [2]."));
}

TEST(type_prop, paged_causal_conv1d_invalid_conv_state_table_rank) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256},  // rank 2 - invalid
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `conv_state_table` input must be one of [3]."));
}

TEST(type_prop, paged_causal_conv1d_invalid_type) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::i32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Element type of `input_embeds` input must be one of [f32, f16, bf16]."));
}

TEST(type_prop, paged_causal_conv1d_hidden_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 128, 4},  // hidden_size 128 != 256
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("hidden_size dimensions of input_embeds and conv_state_table inputs must be "
                                       "compatible"));
}

TEST(type_prop, paged_causal_conv1d_kernel_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 8},  // kernel_size 8 != 4
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("kernel_size dimensions of conv_state_table and conv_weight inputs must be "
                                       "compatible"));
}

TEST(type_prop, paged_causal_conv1d_wrong_input_count) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::PagedCausalConv1D>(OutputVector{p, p, p}),
                    NodeValidationFailure,
                    testing::HasSubstr("PagedCausalConv1D expects 9 inputs. Got: 3"));
}

TEST(type_prop, paged_causal_conv1d_out_channels_hidden_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{128, 256, 4},  // out_channels 128 != hidden_size 256
                                           Shape{128},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("out_channels dimension of conv_weight must be compatible with the hidden_size "
                                       "dimension of input_embeds"));
}

TEST(type_prop, paged_causal_conv1d_conv_bias_size_mismatch) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{128},  // bias size 128 != out_channels 256
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of conv_bias must be compatible with the out_channels dimension of "
                                       "conv_weight or equal to 0 (no bias)"));
}

TEST(type_prop, paged_causal_conv1d_i64_int_inputs_accepted) {
    const auto op = make_pcc(element::f32,
                             Shape{10, 256},
                             Shape{5, 256, 4},
                             Shape{256, 256, 4},
                             Shape{256},
                             Shape{3},
                             Shape{5},
                             Shape{3},
                             Shape{2},
                             Shape{2},
                             element::i64);

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST(type_prop, paged_causal_conv1d_invalid_subsequence_begins_rank) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{2, 2},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("Rank of `subsequence_begins` input must be one of [1]."));
}

TEST(type_prop, paged_causal_conv1d_invalid_subsequence_begins_type) {
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{2},
                                           element::i16),
                    NodeValidationFailure,
                    testing::HasSubstr("Element type of `subsequence_begins` input must be one of [i32, i64]."));
}

TEST(type_prop, paged_causal_conv1d_empty_conv_bias) {
    // conv_bias of shape [0] means no bias - should be valid
    const auto op = make_pcc(element::f32,
                             Shape{10, 256},
                             Shape{5, 256, 4},
                             Shape{256, 256, 4},
                             Shape{0},  // empty bias
                             Shape{3},
                             Shape{5},
                             Shape{3},
                             Shape{2},
                             Shape{2});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST(type_prop, paged_causal_conv1d_dynamic_rank_accepted) {
    // Dynamic rank inputs should be accepted during model import / shape propagation
    const auto op = make_pcc(element::f32,
                             PartialShape::dynamic(),  // input_embeds - dynamic rank
                             PartialShape::dynamic(),  // conv_state_table - dynamic rank
                             PartialShape::dynamic(),  // conv_weight - dynamic rank
                             PartialShape::dynamic(),  // conv_bias - dynamic rank
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1},
                             PartialShape{-1});

    EXPECT_EQ(op->get_output_size(), 1);
}

TEST(type_prop, paged_causal_conv1d_dynamic_type_accepted) {
    // Dynamic element type inputs should be accepted during model import / shape propagation
    const auto op = make_pcc(element::dynamic,
                             Shape{10, 256},
                             Shape{5, 256, 4},
                             Shape{256, 256, 4},
                             Shape{256},
                             Shape{3},
                             Shape{5},
                             Shape{3},
                             Shape{2},
                             Shape{2});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
}

TEST(type_prop, paged_causal_conv1d_la_block_indices_num_blocks_mismatch) {
    // la_block_indices[0] (num_blocks) must match conv_state_table[0]
    // conv_state_table has 5 blocks, but la_block_indices has 10
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},  // num_blocks = 5
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},
                                           Shape{10},  // num_blocks = 10 - mismatch!
                                           Shape{3},
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("num_blocks dimension of la_block_indices must be compatible"));
}

TEST(type_prop, paged_causal_conv1d_subsequence_begins_block_begins_mismatch) {
    // subsequence_begins and la_block_indices_begins must have same size (batch_size_in_sequences + 1)
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},  // subsequence_begins size = 3
                                           Shape{5},
                                           Shape{5},  // la_block_indices_begins size = 5 - mismatch!
                                           Shape{2},
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of subsequence_begins must be compatible with the size of "
                                       "la_block_indices_begins"));
}

TEST(type_prop, paged_causal_conv1d_processed_tokens_size_mismatch) {
    // processed_tokens size must be batch_size_in_sequences (subsequence_begins size - 1)
    // subsequence_begins = 3 means batch_size_in_sequences = 2, so processed_tokens should be 2
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},  // batch_size_in_sequences + 1 = 3
                                           Shape{5},
                                           Shape{3},
                                           Shape{5},  // should be 2, not 5
                                           Shape{2}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of processed_tokens must be batch_size_in_sequences"));
}

TEST(type_prop, paged_causal_conv1d_cache_interval_size_mismatch) {
    // cache_interval size must be batch_size_in_sequences (subsequence_begins size - 1)
    OV_EXPECT_THROW(std::ignore = make_pcc(element::f32,
                                           Shape{10, 256},
                                           Shape{5, 256, 4},
                                           Shape{256, 256, 4},
                                           Shape{256},
                                           Shape{3},  // batch_size_in_sequences + 1 = 3
                                           Shape{5},
                                           Shape{3},
                                           Shape{2},
                                           Shape{5}),  // should be 2, not 5
                    NodeValidationFailure,
                    testing::HasSubstr("size of cache_interval must be batch_size_in_sequences"));
}
}  // namespace ov::test
