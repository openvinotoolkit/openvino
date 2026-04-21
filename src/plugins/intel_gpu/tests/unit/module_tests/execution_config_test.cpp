// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/execution_config.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/subtract.hpp"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

// Build a model with MatMul (compressed weights) + PagedAttention stub.
// PA node triggers is_paged_attention_model detection in apply_model_specific_options.
// PagedAttentionExtension expects exactly 28 inputs (see paged_attention.cpp::validate_and_infer_types):
//  0: query          [rank 2]           f32
//  1: key            [rank 2]           f32
//  2: value          [rank 2]           f32
//  3: key_cache      [rank 2,3,4,5]     f32
//  4: value_cache    [rank 2,3,4,5]     f32
//  5: past_lens      [rank 1]           i32
//  6: subsequence_begins [rank 1]       i32
//  7: block_indices  [rank 1]           i32
//  8: block_indices_begins [rank 1]     i32
//  9: scale          [rank 0]           real
// 10: sliding_window [rank 0]           i32
// 11: alibi_slopes   [rank 1]           real
// 12: max_context_len [rank 0]          i32
// 13: score_aggregation_window [rank 0,1] i32
// 14: rotated_block_indices [rank 1]    i32
// 15: rotation_deltas [rank 1,2]        i32
// 16: rotation_trig_lut [rank 1,2]      f16/f32
// 17: xattention_threshold [rank 1]     f16/f32
// 18: xattention_block_size [rank 0]    i32
// 19: xattention_stride [rank 0]        i32
// 20: sinks          [rank 1,4]         any
// 21: adaptive_rkv_start_size [rank 0]  i32
// 22: adaptive_rkv_evictable_sizes [rank 1] i32
// 23: adaptive_rkv_diversity_block_set_indices [rank 1] i32
// 24: adaptive_rkv_diversity_block_set_indices_begins [rank 1] i32
// 25: token_type_ids [rank 1,2]         i32
// 26: qq_bias        [rank 1]           u8
// 27: qq_bias_begins [rank 1]           i32
static std::shared_ptr<ov::Model> make_pa_matmul_model(ov::element::Type weight_type) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 16});

    auto weight_const = ov::op::v0::Constant::create(weight_type, ov::Shape{32, 16}, {1});

    std::shared_ptr<ov::Node> weight_node;
    if (weight_type == ov::element::u4 || weight_type == ov::element::i4 || weight_type == ov::element::u8) {
        auto convert  = std::make_shared<ov::op::v0::Convert>(weight_const, ov::element::f32);
        auto zp_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {0});
        auto subtract = std::make_shared<ov::op::v1::Subtract>(convert, zp_const);
        auto sc_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, sc_const);
        weight_node = multiply;
    } else {
        weight_node = weight_const;
    }

    auto matmul = std::make_shared<ov::op::v0::MatMul>(input, weight_node, false, true);

    const size_t hs = 64;
    //  0: query
    auto pa_query = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    //  1: key
    auto pa_key = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    //  2: value
    auto pa_value = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, hs});
    //  3: key_cache
    auto pa_key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 16, hs});
    //  4: value_cache
    auto pa_value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1, 16, hs});
    //  5: past_lens
    auto pa_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    //  6: subsequence_begins
    auto pa_subseq_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    //  7: block_indices
    auto pa_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    //  8: block_indices_begins
    auto pa_block_indices_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    //  9: scale
    auto pa_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    // 10: sliding_window
    auto pa_sliding_window = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    // 11: alibi_slopes
    auto pa_alibi_slopes = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    // 12: max_context_len
    auto pa_max_context_len = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    // 13: score_aggregation_window
    auto pa_score_agg_window = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    // 14: rotated_block_indices  (Constant with Shape{0} = no rotation)
    auto pa_rotated_block_indices = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    // 15: rotation_deltas        (Constant with Shape{0} = no rotation)
    auto pa_rotation_deltas = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{0}, {});
    // 16: rotation_trig_lut      (Constant with Shape{0} = no rotation)
    auto pa_rotation_trig_lut = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, {});
    // 17: xattention_threshold
    auto pa_xattn_threshold = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1});
    // 18: xattention_block_size
    auto pa_xattn_block_size = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    // 19: xattention_stride
    auto pa_xattn_stride = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{});
    // 20: sinks
    auto pa_sinks = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 1}, {0});
    // 21: adaptive_rkv_start_size
    auto pa_arkv_start = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    // 22: adaptive_rkv_evictable_sizes
    auto pa_arkv_evict = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    // 23: adaptive_rkv_diversity_block_set_indices
    auto pa_arkv_div_idx = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    // 24: adaptive_rkv_diversity_block_set_indices_begins
    auto pa_arkv_div_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    // 25: token_type_ids
    auto pa_token_type_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{0});
    // 26: qq_bias
    auto pa_qq_bias = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1});
    // 27: qq_bias_begins
    auto pa_qq_bias_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});

    ov::OutputVector pa_args = {
        pa_query, pa_key, pa_value, pa_key_cache, pa_value_cache,       // 0-4
        pa_past_lens, pa_subseq_begins, pa_block_indices,               // 5-7
        pa_block_indices_begins, pa_scale, pa_sliding_window,           // 8-10
        pa_alibi_slopes, pa_max_context_len, pa_score_agg_window,       // 11-13
        pa_rotated_block_indices, pa_rotation_deltas, pa_rotation_trig_lut, // 14-16
        pa_xattn_threshold, pa_xattn_block_size, pa_xattn_stride,      // 17-19
        pa_sinks, pa_arkv_start, pa_arkv_evict,                        // 20-22
        pa_arkv_div_idx, pa_arkv_div_begins, pa_token_type_ids,        // 23-25
        pa_qq_bias, pa_qq_bias_begins                                   // 26-27
    };
    auto pa_node = std::make_shared<ov::op::PagedAttentionExtension>(pa_args);

    return std::make_shared<ov::Model>(
        ov::OutputVector{matmul, pa_node->output(0)},
        ov::ParameterVector{input,
                            pa_query, pa_key, pa_value, pa_key_cache, pa_value_cache,
                            pa_past_lens, pa_subseq_begins, pa_block_indices,
                            pa_block_indices_begins, pa_max_context_len, pa_score_agg_window,
                            pa_xattn_threshold, pa_xattn_block_size, pa_xattn_stride,
                            pa_arkv_evict, pa_arkv_div_idx, pa_arkv_div_begins,
                            pa_token_type_ids, pa_qq_bias, pa_qq_bias_begins});
}

TEST(execution_config, kv_cache_u4_weights_auto_detect_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_i4_weights_auto_detect_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::i4);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    // i4 weights → auto-detect u4, and finalize normalizes i4→u4
    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_u8_weights_no_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u8);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_NE(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_f32_weights_no_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.finalize(ctx.get(), model.get());

    ASSERT_NE(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_user_override_wins) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::i8));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::i8);
}

TEST(execution_config, kv_cache_i4_normalized_to_u4) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::i4));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::u4);
}

TEST(execution_config, kv_cache_u8_normalized_to_i8) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::f32);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::u8));
    config.finalize(ctx.get(), model.get());

    ASSERT_EQ(config.get_kv_cache_precision(), ov::element::i8);
}

TEST(execution_config, kv_cache_4bit_by_token_throws) {
    auto& engine = get_test_engine();
    auto ctx = std::make_shared<RemoteContextImpl>("GPU", std::vector<cldnn::device::ptr>{engine.get_device()});
    auto model = make_pa_matmul_model(ov::element::u4);

    ExecutionConfig config;
    config.set_user_property(ov::hint::kv_cache_precision(ov::element::u4));
    config.set_user_property(ov::internal::key_cache_quant_mode(ov::internal::CacheQuantMode::BY_TOKEN));

    ASSERT_ANY_THROW(config.finalize(ctx.get(), model.get()));
}
