// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/group_query_attention.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/variadic_split.hpp"

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

namespace {
ov::OutputVector make_gqa_args_with_head_sink(const element::Type& sink_type, const PartialShape& sink_shape) {
    const auto empty = op::v0::Constant::create(element::dynamic, Shape{0}, {});
    auto args = make_valid_gqa_args();
    // positions 7-10: cos_cache, sin_cache, position_ids, attention_bias (absent)
    args.insert(args.end(), {empty, empty, empty, empty});
    args.push_back(std::make_shared<op::v0::Parameter>(sink_type, sink_shape));
    return args;
}
}  // namespace

TEST(type_prop, group_query_attention_head_sink_valid) {
    const auto args = make_gqa_args_with_head_sink(element::f32, PartialShape{6});
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_head_sink_dynamic_dim_valid) {
    const auto args = make_gqa_args_with_head_sink(element::f16, PartialShape{-1});
    OV_ASSERT_NO_THROW(std::ignore =
                           std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false));
}

TEST(type_prop, group_query_attention_head_sink_invalid_rank) {
    const auto args = make_gqa_args_with_head_sink(element::f32, PartialShape{1, 6});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Rank of `head_sink` input is not compatible"));
}

TEST(type_prop, group_query_attention_head_sink_invalid_length) {
    const auto args = make_gqa_args_with_head_sink(element::f32, PartialShape{4});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("head_sink must have num_heads (6) elements"));
}

TEST(type_prop, group_query_attention_head_sink_zero_length_invalid) {
    const auto args = make_gqa_args_with_head_sink(element::f32, PartialShape{0});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("head_sink must have num_heads (6) elements"));
}

TEST(type_prop, group_query_attention_head_sink_absent_placeholder_valid) {
    auto args = make_gqa_args_with_head_sink(element::f32, PartialShape{6});
    args.back() = op::v0::Constant::create(element::f32, Shape{0}, {});
    OV_ASSERT_NO_THROW(std::ignore =
                           std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false));
}

TEST(type_prop, group_query_attention_head_sink_invalid_type) {
    const auto args = make_gqa_args_with_head_sink(element::i32, PartialShape{6});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("Element type of `head_sink` input is not compatible"));
}

TEST(type_prop, group_query_attention_static_empty_past_grows_by_current) {
    using ov::op::v0::Parameter;
    auto args = make_valid_gqa_args();
    // A zero-capacity past (absent ONNX past) cannot be written in place: present = current tokens only.
    args[3] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    args[4] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 4, 8}));
}

TEST(type_prop, group_query_attention_static_past_keeps_capacity) {
    const auto op =
        std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(), 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_bf16_activations_and_cache) {
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(element::bf16),
                                                                        6,
                                                                        2,
                                                                        1.0f,
                                                                        false,
                                                                        false);
    EXPECT_EQ(op->get_output_element_type(0), element::bf16);
    EXPECT_EQ(op->get_output_element_type(1), element::bf16);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_rejects_integer_query) {
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(element::i32),
                                                                                      6,
                                                                                      2,
                                                                                      1.0f,
                                                                                      false,
                                                                                      false),
                    ov::NodeValidationFailure,
                    HasSubstr("Element type of `query` input is not compatible"));
}

TEST(type_prop, group_query_attention_shared_kv_present_is_past) {
    using ov::op::v0::Parameter;
    auto args = make_valid_gqa_args();
    // Shared KV: statically empty key/value, nothing appended -> present keeps the past shape (static or dynamic).
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_TRUE(op->is_shared_kv());
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 5, 8}));

    args[3] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    args[4] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, -1, 8}));
}

TEST(type_prop, group_query_attention_regular_kv_is_not_shared) {
    const auto op =
        std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(), 6, 2, 1.0f, false, false);
    EXPECT_FALSE(op->is_shared_kv());
}

TEST(type_prop, group_query_attention_independent_kv_length_separate_kv) {
    using ov::op::v0::Parameter;
    auto args = make_valid_gqa_args();
    EXPECT_FALSE(std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false)
                     ->has_independent_kv_length());

    // A dynamic key length may differ from the query's at runtime (0 for ORT shared KV).
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_TRUE(op->has_independent_kv_length());
    EXPECT_FALSE(op->is_shared_kv());
}

TEST(type_prop, group_query_attention_dynamic_kv_len_static_past_grows_by_key) {
    using ov::op::v0::Parameter;
    auto args = make_valid_gqa_args();
    // Static query and past, dynamic key: no in-place write, present grows by the key's own length.
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, Dimension(5, -1), 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, Dimension(5, -1), 8}));

    // Empty past: present is just the key's length.
    args[3] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    args[4] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, -1, 8}));
}

TEST(type_prop, group_query_attention_independent_kv_length_packed_qkv) {
    using ov::op::v0::Constant;
    using ov::op::v0::Parameter;
    // Packed QKV: Q/K/V split from one tensor, so S_kv == S_q even when the length is dynamic.
    const auto qkv = std::make_shared<Parameter>(element::f32, PartialShape{1, 10, -1, 8});
    const auto split = std::make_shared<ov::op::v1::VariadicSplit>(qkv,
                                                                   Constant::create(element::i64, Shape{}, {1}),
                                                                   Constant::create(element::i64, Shape{3}, {6, 2, 2}));
    auto args = make_valid_gqa_args();
    args[0] = split->output(0);
    args[1] = split->output(1);
    args[2] = split->output(2);
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_FALSE(op->has_independent_kv_length());
}

TEST(type_prop, group_query_attention_independent_kv_length_static_shared_kv) {
    using ov::op::v0::Parameter;
    auto args = make_valid_gqa_args();
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, 0, 8});
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_TRUE(op->is_shared_kv());
    EXPECT_FALSE(op->has_independent_kv_length());
}

namespace {
std::shared_ptr<op::internal::GroupQueryAttention> make_gqa_with_softcap(float softcap) {
    return std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(),
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
                                                               /*causal*/ true,
                                                               softcap);
}
}  // namespace

TEST(type_prop, group_query_attention_softcap_defaults_to_disabled) {
    const auto op =
        std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(), 6, 2, 1.0f, false, false);
    EXPECT_EQ(op->get_softcap(), 0.0f);
}

TEST(type_prop, group_query_attention_softcap_positive_is_valid) {
    const auto op = make_gqa_with_softcap(30.0f);
    EXPECT_EQ(op->get_softcap(), 30.0f);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_softcap_negative_rejected) {
    OV_EXPECT_THROW(std::ignore = make_gqa_with_softcap(-1.0f),
                    ov::NodeValidationFailure,
                    HasSubstr("expects softcap >= 0"));
}

TEST(type_prop, group_query_attention_softcap_nan_rejected) {
    OV_EXPECT_THROW(std::ignore = make_gqa_with_softcap(std::numeric_limits<float>::quiet_NaN()),
                    ov::NodeValidationFailure,
                    HasSubstr("expects softcap >= 0"));
}

}  // namespace testing
}  // namespace ov