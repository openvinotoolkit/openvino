// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

namespace ov {
namespace testing {
using ::testing::HasSubstr;

namespace {
ov::OutputVector make_valid_gqa_args(const element::Type& t = element::f32) {
    using ov::op::v0::Parameter;

    const auto query = std::make_shared<Parameter>(t, PartialShape{1, 6, 4, 8});
    const auto key = std::make_shared<Parameter>(t, PartialShape{1, 2, 4, 8});
    const auto value = std::make_shared<Parameter>(t, PartialShape{1, 2, 4, 8});

    const auto past_key = std::make_shared<Parameter>(t, PartialShape{1, 2, 5, 8});
    const auto past_value = std::make_shared<Parameter>(t, PartialShape{1, 2, 5, 8});

    const auto seqlens_k = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto total_sequence_length = std::make_shared<Parameter>(element::i32, PartialShape{});

    return {query, key, value, past_key, past_value, seqlens_k, total_sequence_length};
}

ov::OutputVector make_valid_gqa_rotary_args(const element::Type& t = element::f32) {
    auto args = make_valid_gqa_args(t);
    const auto cos_cache = std::make_shared<op::v0::Parameter>(t, PartialShape{16, 4});
    const auto sin_cache = std::make_shared<op::v0::Parameter>(t, PartialShape{16, 4});
    args.push_back(cos_cache);
    args.push_back(sin_cache);
    return args;
}

// Fills optional inputs 7-11 with empty constants (treated as absent by has_input).
ov::OutputVector make_valid_gqa_quant_args(const element::Type& kv_type,
                                           int64_t kv_cache_bit_width,
                                           const element::Type& scale_type = element::f32) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    const auto empty = Constant::create(element::dynamic, Shape{0}, {});

    const auto query = std::make_shared<Parameter>(element::f32, PartialShape{1, 6, 4, 8});
    const auto key = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 4, 8});
    const auto value = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 4, 8});
    const auto past_key = std::make_shared<Parameter>(kv_type, PartialShape{1, 2, 5, 8});
    const auto past_value = std::make_shared<Parameter>(kv_type, PartialShape{1, 2, 5, 8});
    const auto seqlens_k = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto total_sequence_length = std::make_shared<Parameter>(element::i32, PartialShape{});
    // positions 7-11: cos_cache, sin_cache, position_ids, attention_mask, head_sink (all absent)
    const auto k_scale = std::make_shared<Parameter>(scale_type, PartialShape{});
    const auto v_scale = std::make_shared<Parameter>(scale_type, PartialShape{});

    return {query,
            key,
            value,
            past_key,
            past_value,
            seqlens_k,
            total_sequence_length,
            empty,
            empty,
            empty,
            empty,
            empty,
            k_scale,
            v_scale};
}
}  // namespace

TEST(type_prop, group_query_attention_gqa_output_shapes) {
    const auto args = make_valid_gqa_args();
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);

    EXPECT_EQ(op->get_output_size(), 3);
    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_element_type(2), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_mha_output_shapes) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 4, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 4, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 4, 8});
    const auto past_key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 5, 8});
    const auto past_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, 5, 8});
    const auto seqlens_k = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    const auto total_sequence_length = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    const auto args = ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 2, 2, 1.0f, false, false);

    EXPECT_EQ(op->get_output_size(), 3);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 16}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_dynamic_seq_len) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 6, -1, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    const auto past_key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    const auto past_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    const auto seqlens_k = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    const auto total_sequence_length = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    const auto args = ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, -1, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, -1, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, -1, 8}));
}

TEST(type_prop, group_query_attention_dynamic_kv_len_accumulates) {
    const auto query = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 6, {1, 4}, 8});
    const auto key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, {1, 4}, 8});
    const auto value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, {1, 4}, 8});
    const auto past_key = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, {5, 9}, 8});
    const auto past_value = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 2, {5, 9}, 8});
    const auto seqlens_k = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{1});
    const auto total_sequence_length = std::make_shared<op::v0::Parameter>(element::i32, PartialShape{});

    const auto args = ov::OutputVector{query, key, value, past_key, past_value, seqlens_k, total_sequence_length};

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, {1, 4}, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, {6, 13}, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, {6, 13}, 8}));
}

TEST(type_prop, group_query_attention_invalid_query_rank) {
    auto args = make_valid_gqa_args();
    args[0] = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{1, 6, 8});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Rank of `query` input"));
}

TEST(type_prop, group_query_attention_do_rotary_requires_cos_sin) {
    const auto args = make_valid_gqa_args();

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, true, false),
                    ov::NodeValidationFailure,
                    HasSubstr("cos_cache"));
}

TEST(type_prop, group_query_attention_rotary_inputs_static_shapes) {
    const auto args = make_valid_gqa_rotary_args();
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, true, false);

    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_partial_rotary_dim_accepted) {
    // head_size == 8; cos_cache last dim == 2 -> rotary_dim == 4 < head_size (GPT-NeoX/Phi-style partial RoPE).
    auto args = make_valid_gqa_args();
    const auto cos_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, 2});
    const auto sin_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, 2});
    args.push_back(cos_cache);
    args.push_back(sin_cache);

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, true, false);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_rotary_dim_exceeds_head_size) {
    // head_size == 8; cos_cache last dim == 8 -> rotary_dim == 16 > head_size.
    auto args = make_valid_gqa_args();
    const auto cos_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, 8});
    const auto sin_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, 8});
    args.push_back(cos_cache);
    args.push_back(sin_cache);

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, true, false),
                    ov::NodeValidationFailure,
                    HasSubstr("must not exceed head_size"));
}

TEST(type_prop, group_query_attention_rotary_dim_dynamic_cos_with_static_head_size) {
    auto args = make_valid_gqa_args();
    const auto cos_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, -1});
    const auto sin_cache = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{16, -1});
    args.push_back(cos_cache);
    args.push_back(sin_cache);

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, true, false),
                    ov::NodeValidationFailure,
                    HasSubstr("must be statically known"));
}

TEST(type_prop, group_query_attention_quant_type_enum_names) {
    EXPECT_EQ(as_string(op::internal::GroupQueryAttentionQuantType::NONE), "NONE");
    EXPECT_EQ(as_string(op::internal::GroupQueryAttentionQuantType::PER_TENSOR), "PER_TENSOR");
    EXPECT_EQ(as_string(op::internal::GroupQueryAttentionQuantType::PER_CHANNEL), "PER_CHANNEL");
    EXPECT_EQ(as_enum<op::internal::GroupQueryAttentionQuantType>("PER_TENSOR"),
              op::internal::GroupQueryAttentionQuantType::PER_TENSOR);
}

// ---------- quantized KV cache ----------

TEST(type_prop, group_query_attention_kv_cache_int8_per_tensor) {
    const auto args = make_valid_gqa_quant_args(element::i8, 8);
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args,
                                                                        6,
                                                                        2,
                                                                        1.0f,
                                                                        false,
                                                                        false,
                                                                        8,
                                                                        quantize_type,
                                                                        quantize_type);

    EXPECT_EQ(op->get_output_element_type(0), element::f32);
    EXPECT_EQ(op->get_output_element_type(1), element::i8);
    EXPECT_EQ(op->get_output_element_type(2), element::i8);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_kv_cache_uint4_not_supported) {
    // u4 is not in the allowed past_key/past_value type list; remove this when u4 support is added.
    const auto args = make_valid_gqa_quant_args(element::u4, 4);
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;
    OV_EXPECT_THROW(
        std::ignore = std::make_shared<
            op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false, 4, quantize_type, quantize_type),
        ov::NodeValidationFailure,
        HasSubstr("past_key"));
}

TEST(type_prop, group_query_attention_kv_cache_mismatched_quant_types) {
    const auto args = make_valid_gqa_quant_args(element::i8, 8);
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(
                        args,
                        6,
                        2,
                        1.0f,
                        false,
                        false,
                        8,
                        op::internal::GroupQueryAttentionQuantType::PER_TENSOR,
                        op::internal::GroupQueryAttentionQuantType::PER_CHANNEL),
                    ov::NodeValidationFailure,
                    HasSubstr("matching k_quant_type and v_quant_type"));
}

TEST(type_prop, group_query_attention_kv_cache_mismatched_past_types) {
    auto args = make_valid_gqa_quant_args(element::i8, 8);
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_VALUE)] =
        std::make_shared<op::v0::Parameter>(element::u8, PartialShape{1, 2, 5, 8});
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<
            op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false, 8, quantize_type, quantize_type),
        ov::NodeValidationFailure,
        HasSubstr("past_key and past_value element types to match"));
}

TEST(type_prop, group_query_attention_quantized_kv_requires_quantized_cache_type) {
    const auto args = make_valid_gqa_quant_args(element::f32, 8);
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<
            op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false, 8, quantize_type, quantize_type),
        ov::NodeValidationFailure,
        HasSubstr("quantized KV cache element type"));
}

// ---------- causal ----------

TEST(type_prop, group_query_attention_causal_defaults_true) {
    const auto args = make_valid_gqa_args();
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_TRUE(op->get_causal());
}

TEST(type_prop, group_query_attention_bidirectional_without_window_is_valid) {
    const auto args = make_valid_gqa_args();
    const auto op =
        std::make_shared<op::internal::GroupQueryAttention>(args,
                                                            6,
                                                            2,
                                                            1.0f,
                                                            false,
                                                            false,
                                                            /*kv_cache_bit_width*/ 0,
                                                            op::internal::GroupQueryAttentionQuantType::NONE,
                                                            op::internal::GroupQueryAttentionQuantType::NONE,
                                                            /*local_window_size*/ -1,
                                                            /*sliding_window_cache*/ false,
                                                            /*smooth_softmax*/ false,
                                                            /*causal*/ false);
    EXPECT_FALSE(op->get_causal());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_causal_false_rejects_window) {
    const auto args = make_valid_gqa_args();
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(
                        args,
                        6,
                        2,
                        1.0f,
                        false,
                        false,
                        /*kv_cache_bit_width*/ 0,
                        op::internal::GroupQueryAttentionQuantType::NONE,
                        op::internal::GroupQueryAttentionQuantType::NONE,
                        /*local_window_size*/ 2,
                        /*sliding_window_cache*/ false,
                        /*smooth_softmax*/ false,
                        /*causal*/ false),
                    ov::NodeValidationFailure,
                    HasSubstr("local_window_size requires causal=1"));
}

}  // namespace testing
}  // namespace ov