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

TEST(type_prop, group_query_attention_absent_past_is_empty_cache) {
    auto args = make_valid_gqa_args();
    const auto empty = op::v0::Constant::create(element::f32, Shape{1, 2, 0, 8}, {});
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_KEY)] = empty;
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_VALUE)] = empty;

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);

    EXPECT_EQ(op->get_output_element_type(1), element::f32);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 4, 8}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 4, 8}));
}

TEST(type_prop, group_query_attention_absent_past_key_only) {
    auto args = make_valid_gqa_args();
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_KEY)] =
        op::v0::Constant::create(element::f32, Shape{1, 2, 0, 8}, {});

    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("both present or both absent"));
}

TEST(type_prop, group_query_attention_absent_typed_past_with_quantized_cache) {
    auto args = make_valid_gqa_quant_args(element::i8, 8);
    const auto empty = op::v0::Constant::create(element::i8, Shape{1, 2, 0, 8}, {});
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_KEY)] = empty;
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_VALUE)] = empty;
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

    EXPECT_EQ(op->get_output_element_type(1), element::i8);
    EXPECT_EQ(op->get_output_element_type(2), element::i8);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 4, 8}));
}

TEST(type_prop, group_query_attention_absent_typed_past_with_4bit_cache) {
    auto args = make_valid_gqa_quant_args(element::u8, 4);
    const auto empty = op::v0::Constant::create(element::u8, Shape{1, 2, 0, 4}, {});
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_KEY)] = empty;
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_VALUE)] = empty;
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;

    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args,
                                                                        6,
                                                                        2,
                                                                        1.0f,
                                                                        false,
                                                                        false,
                                                                        4,
                                                                        quantize_type,
                                                                        quantize_type);

    EXPECT_EQ(op->get_output_element_type(1), element::u8);
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{1, 2, 4, 4}));
    EXPECT_EQ(op->get_output_partial_shape(2), (PartialShape{1, 2, 4, 4}));
}

TEST(type_prop, group_query_attention_absent_unshaped_past_with_quantized_cache) {
    auto args = make_valid_gqa_quant_args(element::i8, 8);
    const auto empty = op::v0::Constant::create(element::i8, Shape{0}, {});
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_KEY)] = empty;
    args[static_cast<size_t>(op::internal::GroupQueryAttentionInputs::PAST_VALUE)] = empty;
    const auto quantize_type = op::internal::GroupQueryAttentionQuantType::PER_TENSOR;

    OV_EXPECT_THROW(
        std::ignore = std::make_shared<
            op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false, 8, quantize_type, quantize_type),
        ov::NodeValidationFailure,
        HasSubstr("typed rank-4 empty past"));
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

namespace {
// Valid base args (head_size 8) with optional inputs 7-13 empty and QK-Norm weights at 14/15.
ov::OutputVector make_gqa_args_with_qk_norm(const ov::PartialShape& q_weight_shape,
                                            const ov::PartialShape& k_weight_shape,
                                            const element::Type& weight_type = element::f32,
                                            bool with_k_weight = true) {
    auto args = make_valid_gqa_args();
    const auto empty = op::v0::Constant::create(element::dynamic, Shape{0}, {});
    args.resize(14, empty);
    args.push_back(std::make_shared<op::v0::Parameter>(weight_type, q_weight_shape));
    args.push_back(with_k_weight ? std::make_shared<op::v0::Parameter>(weight_type, k_weight_shape)->output(0)
                                 : empty->output(0));
    return args;
}

std::shared_ptr<op::internal::GroupQueryAttention> make_gqa_with_qk_norm(const ov::OutputVector& args,
                                                                         float epsilon = 1e-6f) {
    return std::make_shared<op::internal::GroupQueryAttention>(args,
                                                               6,
                                                               2,
                                                               1.0f,
                                                               false,
                                                               false,
                                                               0,
                                                               op::internal::GroupQueryAttentionQuantType::NONE,
                                                               op::internal::GroupQueryAttentionQuantType::NONE,
                                                               -1,
                                                               false,
                                                               false,
                                                               true,
                                                               0.0f,
                                                               epsilon);
}
}  // namespace

TEST(type_prop, group_query_attention_qk_norm_valid) {
    const auto op = make_gqa_with_qk_norm(make_gqa_args_with_qk_norm(PartialShape{8}, PartialShape{8}), 1e-5f);
    EXPECT_TRUE(op->has_qk_norm());
    EXPECT_EQ(op->get_qk_norm_epsilon(), 1e-5f);
    EXPECT_EQ(op->get_output_partial_shape(0), (PartialShape{1, 4, 48}));
}

TEST(type_prop, group_query_attention_qk_norm_absent_by_default) {
    const auto op =
        std::make_shared<op::internal::GroupQueryAttention>(make_valid_gqa_args(), 6, 2, 1.0f, false, false);
    EXPECT_FALSE(op->has_qk_norm());
    EXPECT_EQ(op->get_qk_norm_epsilon(), 1e-6f);
}

TEST(type_prop, group_query_attention_qk_norm_dynamic_weight_valid) {
    OV_ASSERT_NO_THROW(std::ignore = make_gqa_with_qk_norm(
                           make_gqa_args_with_qk_norm(PartialShape::dynamic(1), PartialShape::dynamic())));
}

TEST(type_prop, group_query_attention_qk_norm_unpaired_rejected) {
    const auto args = make_gqa_args_with_qk_norm(PartialShape{8}, PartialShape{8}, element::f32, false);
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(args),
                    ov::NodeValidationFailure,
                    HasSubstr("q_norm_weight and k_norm_weight must be provided together"));
}

TEST(type_prop, group_query_attention_qk_norm_wrong_length_rejected) {
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(make_gqa_args_with_qk_norm(PartialShape{8}, PartialShape{4})),
                    ov::NodeValidationFailure,
                    HasSubstr("k_norm_weight must have head_size (8) elements"));
}

TEST(type_prop, group_query_attention_qk_norm_wrong_rank_rejected) {
    OV_EXPECT_THROW(
        std::ignore = make_gqa_with_qk_norm(make_gqa_args_with_qk_norm(PartialShape{1, 8}, PartialShape{8})),
        ov::NodeValidationFailure,
        HasSubstr("Rank of `q_norm_weight` input is not compatible"));
}

TEST(type_prop, group_query_attention_qk_norm_type_mismatch_rejected) {
    OV_EXPECT_THROW(
        std::ignore = make_gqa_with_qk_norm(make_gqa_args_with_qk_norm(PartialShape{8}, PartialShape{8}, element::f16)),
        ov::NodeValidationFailure,
        HasSubstr("q_norm_weight element type must match query"));
}

TEST(type_prop, group_query_attention_qk_norm_non_positive_epsilon_rejected) {
    const auto args = make_gqa_args_with_qk_norm(PartialShape{8}, PartialShape{8});
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(args, 0.0f),
                    ov::NodeValidationFailure,
                    HasSubstr("finite qk_norm_epsilon > 0"));
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(args, std::numeric_limits<float>::infinity()),
                    ov::NodeValidationFailure,
                    HasSubstr("finite qk_norm_epsilon > 0"));
}

TEST(type_prop, group_query_attention_non_positive_epsilon_rejected_without_qk_norm) {
    const auto args = make_valid_gqa_args();
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(args, 0.0f),
                    ov::NodeValidationFailure,
                    HasSubstr("finite qk_norm_epsilon > 0"));
    OV_EXPECT_THROW(std::ignore = make_gqa_with_qk_norm(args, std::numeric_limits<float>::quiet_NaN()),
                    ov::NodeValidationFailure,
                    HasSubstr("finite qk_norm_epsilon > 0"));
}

namespace {
std::shared_ptr<op::internal::GroupQueryAttention> make_gqa_with_qk_output(
    op::internal::GroupQueryAttentionQKOutput qk_output,
    bool sliding_window_cache = false,
    const ov::OutputVector& args = make_valid_gqa_args()) {
    return std::make_shared<op::internal::GroupQueryAttention>(args,
                                                               6,
                                                               2,
                                                               1.0f,
                                                               false,
                                                               false,
                                                               0,
                                                               op::internal::GroupQueryAttentionQuantType::NONE,
                                                               op::internal::GroupQueryAttentionQuantType::NONE,
                                                               sliding_window_cache ? 4 : -1,
                                                               sliding_window_cache,
                                                               false,
                                                               true,
                                                               0.0f,
                                                               1e-6f,
                                                               qk_output);
}
}  // namespace

TEST(type_prop, group_query_attention_qk_output_none_has_three_outputs) {
    const auto op = make_gqa_with_qk_output(op::internal::GroupQueryAttentionQKOutput::NONE);
    EXPECT_EQ(op->get_output_size(), 3);
}

TEST(type_prop, group_query_attention_qk_output_adds_scores_output) {
    for (const auto mode : {op::internal::GroupQueryAttentionQKOutput::BEFORE_SOFTMAX,
                            op::internal::GroupQueryAttentionQKOutput::AFTER_SOFTMAX}) {
        const auto op = make_gqa_with_qk_output(mode);
        ASSERT_EQ(op->get_output_size(), 4);
        EXPECT_EQ(op->get_qk_output(), mode);
        EXPECT_EQ(op->get_output_element_type(3), element::f32);
        EXPECT_EQ(op->get_output_partial_shape(3), (PartialShape{1, 6, 4, Dimension::dynamic()}));
    }
}

// output_qk is laid out over total_sequence_length, so qk_output needs that input.
TEST(type_prop, group_query_attention_qk_output_requires_total_sequence_length) {
    auto args = make_valid_gqa_args();
    args[6] = op::v0::Constant::create(element::dynamic, Shape{0}, {});
    OV_EXPECT_THROW(
        std::ignore = make_gqa_with_qk_output(op::internal::GroupQueryAttentionQKOutput::BEFORE_SOFTMAX, false, args),
        ov::NodeValidationFailure,
        HasSubstr("requires total_sequence_length"));
}

TEST(type_prop, group_query_attention_qk_output_with_sliding_window_cache_rejected) {
    OV_EXPECT_THROW(
        std::ignore = make_gqa_with_qk_output(op::internal::GroupQueryAttentionQKOutput::AFTER_SOFTMAX, true),
        ov::NodeValidationFailure,
        HasSubstr("qk_output is not supported with sliding_window_cache"));
}

TEST(type_prop, group_query_attention_fixed_capacity_with_dynamic_batch) {
    // A preallocated cache keeps its capacity whatever the batch: present stays [?, 2, 5, 8].
    using op::v0::Parameter;
    auto args = make_valid_gqa_args();
    args[0] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 6, 4, 8});
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 2, 4, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 2, 4, 8});
    args[3] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 2, 5, 8});
    args[4] = std::make_shared<Parameter>(element::f32, PartialShape{-1, 2, 5, 8});
    const auto op = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false);
    EXPECT_TRUE(op->has_fixed_kv_capacity());
    EXPECT_EQ(op->get_output_partial_shape(1), (PartialShape{-1, 2, 5, 8}));
}

TEST(type_prop, group_query_attention_growing_cache_has_no_fixed_capacity) {
    using op::v0::Parameter;
    auto dynamic_past = make_valid_gqa_args();
    dynamic_past[3] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    dynamic_past[4] = std::make_shared<Parameter>(element::f32, PartialShape{1, 2, -1, 8});
    EXPECT_FALSE(std::make_shared<op::internal::GroupQueryAttention>(dynamic_past, 6, 2, 1.0f, false, false)
                     ->has_fixed_kv_capacity());
    auto dynamic_step = make_valid_gqa_args();
    dynamic_step[0] = std::make_shared<Parameter>(element::f32, PartialShape{1, 6, -1, 8});
    EXPECT_FALSE(std::make_shared<op::internal::GroupQueryAttention>(dynamic_step, 6, 2, 1.0f, false, false)
                     ->has_fixed_kv_capacity());
}

TEST(type_prop, group_query_attention_seqlens_k_must_match_batch) {
    // ORT: seqlens_k holds batch_size elements; a single length must not silently cover a batch of 2.
    using op::v0::Parameter;
    auto args = make_valid_gqa_args();
    args[0] = std::make_shared<Parameter>(element::f32, PartialShape{2, 6, 4, 8});
    args[1] = std::make_shared<Parameter>(element::f32, PartialShape{2, 2, 4, 8});
    args[2] = std::make_shared<Parameter>(element::f32, PartialShape{2, 2, 4, 8});
    args[3] = std::make_shared<Parameter>(element::f32, PartialShape{2, 2, 5, 8});
    args[4] = std::make_shared<Parameter>(element::f32, PartialShape{2, 2, 5, 8});
    args[5] = std::make_shared<Parameter>(element::i32, PartialShape{1, 1});
    OV_EXPECT_THROW(std::ignore = std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false),
                    ov::NodeValidationFailure,
                    HasSubstr("seqlens_k must have batch_size (2) elements"));
    args[5] = std::make_shared<Parameter>(element::i32, PartialShape{2, 1});
    OV_ASSERT_NO_THROW(std::ignore =
                           std::make_shared<op::internal::GroupQueryAttention>(args, 6, 2, 1.0f, false, false));
}

}  // namespace testing
}  // namespace ov