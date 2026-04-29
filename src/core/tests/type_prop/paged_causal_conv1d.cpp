// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_causal_conv1d.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/type_prop.hpp"
#include "openvino/openvino.hpp"

namespace ov::test {

class TypePropPagedCausalConv1DTest : public TypePropOpTest<op::internal::PagedCausalConv1D> {};

TEST_F(TypePropPagedCausalConv1DTest, f32_static_shapes) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST_F(TypePropPagedCausalConv1DTest, f16_static_shapes) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f16, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f16, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f16, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f16, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::f16);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST_F(TypePropPagedCausalConv1DTest, bf16_partial_shapes) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {128, 512}});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, {128, 512}, -1});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1, -1, -1});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::bf16, PartialShape{-1});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{-1, {128, 512}}));
}

TEST_F(TypePropPagedCausalConv1DTest, invalid_input_embeds_rank) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256, 1});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[0].rank().compatible(2)"));
}

TEST_F(TypePropPagedCausalConv1DTest, invalid_conv_state_table_rank) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[1].rank().compatible(3)"));
}

TEST_F(TypePropPagedCausalConv1DTest, invalid_float_type) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::i32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::i32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::i32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::i32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("Float inputs must have f32, f16, or bf16 element type."));
}

TEST_F(TypePropPagedCausalConv1DTest, state_float_type_independent_from_embeds) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f16, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST_F(TypePropPagedCausalConv1DTest, float_type_mismatch_among_embeds_weight_bias) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f16, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("PagedCausalConv1D expects input_embeds, conv_weight, and conv_bias to have "
                                       "the same element type."));
}

TEST_F(TypePropPagedCausalConv1DTest, hidden_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 128, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("hidden_size dimensions of input_embeds and conv_state_table inputs must be "
                                       "compatible"));
}

TEST_F(TypePropPagedCausalConv1DTest, kernel_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 8});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("kernel_size dimensions of conv_state_table and conv_weight inputs must be "
                                       "compatible"));
}

TEST_F(TypePropPagedCausalConv1DTest, wrong_input_count) {
    auto p = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{p, p, p}),
                    NodeValidationFailure,
                    testing::HasSubstr("Check 'get_input_size() == 9'"));
}

TEST_F(TypePropPagedCausalConv1DTest, out_channels_hidden_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{128, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{128});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("out_channels dimension of conv_weight must be compatible with the hidden_size "
                                       "dimension of input_embeds"));
}

TEST_F(TypePropPagedCausalConv1DTest, conv_bias_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{128});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of conv_bias must be compatible with the out_channels dimension of "
                                       "conv_weight or equal to 0 (no bias)"));
}

TEST_F(TypePropPagedCausalConv1DTest, i64_integer_inputs_accepted) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i64, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i64, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i64, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST_F(TypePropPagedCausalConv1DTest, invalid_subsequence_begins_rank) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{2, 2});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("input_shapes[4].rank().compatible(1)"));
}

TEST_F(TypePropPagedCausalConv1DTest, invalid_integer_type) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i16, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i16, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i16, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i16, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i16, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("Integer inputs must have i32 or i64 element type."));
}

TEST_F(TypePropPagedCausalConv1DTest, empty_conv_bias) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{0});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), PartialShape(Shape{10, 256}));
}

TEST_F(TypePropPagedCausalConv1DTest, dynamic_rank_accepted) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, PartialShape::dynamic());
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{-1});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
}

TEST_F(TypePropPagedCausalConv1DTest, dynamic_type_accepted) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::dynamic, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    const auto op = make_op(OutputVector{input_embeds,
                                         conv_state_table,
                                         conv_weight,
                                         conv_bias,
                                         subsequence_begins,
                                         la_block_indices,
                                         la_block_indices_begins,
                                         processed_tokens,
                                         cache_interval});

    EXPECT_EQ(op->get_output_size(), 1);
    EXPECT_EQ(op->get_output_element_type(0), element::dynamic);
}

TEST_F(TypePropPagedCausalConv1DTest, la_block_indices_num_blocks_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{10});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("num_blocks dimension of la_block_indices must be compatible"));
}

TEST_F(TypePropPagedCausalConv1DTest, subsequence_begins_block_begins_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of subsequence_begins must be compatible with the size of "
                                       "la_block_indices_begins"));
}

TEST_F(TypePropPagedCausalConv1DTest, processed_tokens_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of processed_tokens must be batch_size_in_sequences"));
}

TEST_F(TypePropPagedCausalConv1DTest, cache_interval_size_mismatch) {
    const auto input_embeds = std::make_shared<op::v0::Parameter>(element::f32, Shape{10, 256});
    const auto conv_state_table = std::make_shared<op::v0::Parameter>(element::f32, Shape{5, 256, 4});
    const auto conv_weight = std::make_shared<op::v0::Parameter>(element::f32, Shape{256, 256, 4});
    const auto conv_bias = std::make_shared<op::v0::Parameter>(element::f32, Shape{256});
    const auto subsequence_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto la_block_indices = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});
    const auto la_block_indices_begins = std::make_shared<op::v0::Parameter>(element::i32, Shape{3});
    const auto processed_tokens = std::make_shared<op::v0::Parameter>(element::i32, Shape{2});
    const auto cache_interval = std::make_shared<op::v0::Parameter>(element::i32, Shape{5});

    OV_EXPECT_THROW(std::ignore = make_op(OutputVector{input_embeds,
                                                       conv_state_table,
                                                       conv_weight,
                                                       conv_bias,
                                                       subsequence_begins,
                                                       la_block_indices,
                                                       la_block_indices_begins,
                                                       processed_tokens,
                                                       cache_interval}),
                    NodeValidationFailure,
                    testing::HasSubstr("size of cache_interval must be batch_size_in_sequences"));
}
}  // namespace ov::test
