// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>
#include <openvino/core/except.hpp>
#include <openvino/reference/adaptive_rkv_diversity.hpp>
#include <openvino/reference/xattention.hpp>
#include <optional>

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "primitive_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

// Enable detailed xattention debugging (dumps, extra comparison info)
// Default: OFF (0). Set to 1 for investigation.
#ifndef XATTENTION_DEBUG_VERBOSE
#    define XATTENTION_DEBUG_VERBOSE 0
#endif

/*
 * PagedAttention inputs:
 * [0]: query, shape: [batch_size_in_tokens, num_heads * head_size], type: f16
 * [1]: key, shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
 * [2]: value, shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
 * [3]: key_cache, shape: [num_blocks, num_kv_heads, head_size, block_size], type: f16 or i8
 * [4]: value_cache, shape: [num_blocks, num_kv_heads, block_size, head_size], type: f16 or i8
 * [5]: past_lens, shape: [batch_size_in_sequences], type: i32
 * [6]: subsequence_begins, shape: [batch_size_in_sequences + 1], type: i32
 * [7]: block_indices, shape: [num_blocks], type: i32
 * [8]: block_indices_begins, shape: [batch_size_in_sequences + 1], type: i32
 * [9]: scale, optional
 * [10]: sliding_window, optional
 * [11]: alibi_slopes, optional
 * [12]: max_context_len, shape: [], type: i32
 * [13]: score_aggregation_window, optional, shape: [batch_size_in_sequences], type: i32
 * [14]: rotated_block_indices, optional, shape: [num_rotated_blocks], type: i32
 * [15]: rotation_deltas, optional, shape: [num_rotated_blocks, BLOCK_SIZE] or [num_rotated_blocks, 1], type: i32
 * [16]: rotation_trig_lut, optional, shape: [max_num_batched_tokens, head_size], type: f16
 * [17]: adaptive_rkv_start_size, optional, shape: [], type: i32
 * [18]: adaptive_rkv_evictable_sizes, optional, shape: [batch_size_in_sequences], type: i32
 * [19]: adaptive_rkv_diversity_block_set_indices, optional, shape: [total_blocks], type: i32
 * [20]: adaptive_rkv_diversity_block_set_indices_begins, optional, shape: [batch_size_in_sequences + 1], type: i32
 * [21]: qq_bias, optional, shape: [total_mask_size], type: u8
 * [22]: qq_bias_begins, optional, shape: [batch_size_in_sequences + 1], type: i32
 */

enum class ScoresMode { DISABLED = 0, LAST_TOKEN, SNAPKV };

struct SubsequenceDescriptor {
    int num_tokens;
    int past_len;
};

struct CacheRotationDescriptor {
    bool apply_rotation;
    // configures 2nd dimension of rotation_deltas
    // if per_block is true, single value is used for all tokens inside the block
    // otherwise, each token uses an independent value
    bool per_block;
};

struct QueryToQueryAttentionDescriptor {
    std::vector<std::vector<uint8_t>> qq_bias;
    std::vector<int> qq_bias_begins;
};

struct PagedAttentionManager {
    int num_heads;
    int num_kv_heads;
    int k_head_size;
    int v_head_size;
    int block_size;
    int sliding_window_size;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    ov::element::Type kv_cache_precision = ov::element::dynamic;
    bool has_score_aggregation;
    CacheRotationDescriptor rotation_config;
    std::vector<SubsequenceDescriptor> subsequence_descs;

    // per-subsequence QKV inputs
    std::vector<std::vector<ov::float16>> query_data;  // {[1, num_tokens, num_heads, k_head_size], ..}
    std::vector<std::vector<ov::float16>> key_data;    // {[1, past_len + num_tokens, num_heads, k_head_size], ..}
    std::vector<std::vector<ov::float16>> value_data;  // {[1, past_len + num_tokens, num_heads, v_head_size], ..}

    // common PA inputs
    std::vector<int> past_lens;
    std::vector<int> subsequence_begins;
    std::vector<int> block_indices;
    std::vector<int> block_indices_begins;
    std::vector<int> max_context_len;
    std::vector<int> score_aggregation_window;

    // score aggregation related inputs
    std::vector<int> score_aggregation;

    // rotation related inputs
    std::vector<int> rotated_block_indices;
    std::vector<int> rotation_deltas;
    std::vector<ov::float16> rotation_trig_lut;

    // xattention related inputs
    bool has_xattention;
    std::vector<ov::float16> xattention_threshold;
    std::vector<int> xattention_block_size;
    std::vector<int> xattention_stride;

    std::vector<ov::float16> sinks;

    int adaptive_rkv_start_size = 0;
    std::vector<int> adaptive_rkv_evictable_sizes;
    std::vector<int> adaptive_rkv_diversity_block_set_indices;
    std::vector<int> adaptive_rkv_diversity_block_set_indices_begins;

    std::vector<std::vector<uint8_t>> qq_bias;
    std::vector<int> qq_bias_begins;

    // optional token_type_ids; when empty, a default single-element {0} buffer is used
    std::vector<int> token_type_ids;
    cldnn::engine& test_engine;
    cldnn::stream& test_stream;
    tests::random_generator& rg;

    PagedAttentionManager(tests::random_generator& rg,
                          cldnn::engine& engine,
                          cldnn::stream& stream,
                          const std::vector<SubsequenceDescriptor>& subsequence_descs,
                          int num_heads,
                          int num_kv_heads,
                          int k_head_size,
                          int v_head_size,
                          int block_size,
                          int sliding_window_size,
                          bool kv_cache_compression,
                          ov::internal::CacheQuantMode key_cache_quant_mode,
                          bool has_score_aggregation,
                          bool has_xattention,
                          CacheRotationDescriptor rotation_config,
                          ov::element::Type kv_cache_precision = ov::element::dynamic)
        : num_heads(num_heads),
          num_kv_heads(num_kv_heads),
          k_head_size(k_head_size),
          v_head_size(v_head_size),
          block_size(block_size),
          sliding_window_size(sliding_window_size),
          kv_cache_compression(kv_cache_compression),
          key_cache_quant_mode(key_cache_quant_mode),
          kv_cache_precision(kv_cache_precision),
          has_score_aggregation(has_score_aggregation),
          rotation_config(rotation_config),
          subsequence_descs(subsequence_descs),
          has_xattention(has_xattention),
          test_engine(engine),
          test_stream(stream),
          rg(rg) {
        // init subsequence_begins and block_indices_begins
        subsequence_begins.push_back(0);
        block_indices_begins.push_back(0);

        int max_len = 0;
        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const auto& subsequence_desc = subsequence_descs[i];
            max_len = std::max(max_len, subsequence_desc.num_tokens + subsequence_desc.past_len);

            query_data.push_back(generate_realistic_data(num_heads, subsequence_desc.num_tokens, k_head_size));
            key_data.push_back(generate_realistic_data(num_kv_heads, subsequence_desc.num_tokens + subsequence_desc.past_len, k_head_size));
            value_data.push_back(generate_realistic_data(num_kv_heads, subsequence_desc.num_tokens + subsequence_desc.past_len, v_head_size));

            past_lens.push_back(subsequence_desc.past_len);
            int subsequence_start_pos = subsequence_begins[i];
            int subsequence_end_pos = subsequence_start_pos + subsequence_desc.num_tokens;
            subsequence_begins.push_back(subsequence_end_pos);

            int subsequence_length = subsequence_desc.num_tokens + subsequence_desc.past_len;
            int required_blocks = cldnn::ceil_div(subsequence_length, block_size);
            int start_block_idx = block_indices.empty() ? 0 : block_indices.back() + 1;
            int end_block_idx = start_block_idx + required_blocks;
            for (int block_idx = start_block_idx; block_idx < end_block_idx; block_idx++) {
                block_indices.push_back(block_idx);
            }

            int block_indices_start_pos = block_indices_begins[i];
            int block_indices_end_pos = block_indices_start_pos + required_blocks;
            block_indices_begins.push_back(block_indices_end_pos);
        }
        max_context_len.push_back(max_len);

        if (rotation_config.apply_rotation) {
            // iterate over KV-cache blocks and apply cache rotation to every second
            // fully occupied block
            for (size_t i = 0; i < subsequence_descs.size(); i++) {
                const auto& subsequence_desc = subsequence_descs[i];
                int past_len = subsequence_desc.past_len;
                int start_block_idx = block_indices_begins[i];
                for (int block_idx = 1; block_idx < past_len / block_size; block_idx++) {
                    if (block_idx % 2 != 0) {
                        rotated_block_indices.push_back(start_block_idx + block_idx);
                    }
                }
            }

            if (!rotated_block_indices.empty()) {
                rotation_deltas = generate_rotation_deltas_data(rg, max_context_len[0], rotated_block_indices.size(), block_size, rotation_config.per_block);
                rotation_trig_lut = generate_rotation_trig_lut_data(rg, max_context_len[0], k_head_size);
            }
        }

        if (has_score_aggregation) {
            for (const auto& subsequence_desc : subsequence_descs) {
                const auto max_tokens = 10;
                auto max_window_size = std::min(subsequence_desc.num_tokens, max_tokens);
                auto window_size = rg.generate_random_val<int>(1, max_window_size);
                score_aggregation.push_back(window_size);
            }
        }
    }

    cldnn::memory::ptr get_query_memory() {
        return get_QKV_memory(query_data, num_heads, k_head_size, false);
    }

    cldnn::memory::ptr get_key_memory() {
        return get_QKV_memory(key_data, num_kv_heads, k_head_size, true);
    }

    cldnn::memory::ptr get_value_memory() {
        return get_QKV_memory(value_data, num_kv_heads, v_head_size, true);
    }

    cldnn::memory::ptr get_key_cache_memory_cm() {
        constexpr int kv_sub_block_size = 16;
        auto key_cache_dt = kv_cache_compression ? cldnn::data_types::i8 : cldnn::data_types::f16;
        const int head_size = k_head_size;
        int adjusted_head_size = head_size;
        int adjusted_block_size = block_size;
        if (kv_cache_compression) {
            if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                OPENVINO_ASSERT(block_size % kv_sub_block_size == 0);
                adjusted_block_size += block_size / kv_sub_block_size * 4;
            } else {
                adjusted_head_size += 4;
            }
        }

        const auto num_blocks = block_indices.back() + 1;
        auto key_cache_shape = ov::PartialShape{static_cast<int64_t>(num_blocks),
                                                static_cast<int64_t>(num_kv_heads),
                                                static_cast<int64_t>(adjusted_block_size),
                                                static_cast<int64_t>(adjusted_head_size)};
        auto key_cache_layout = cldnn::layout{key_cache_shape, key_cache_dt, cldnn::format::bfyx};
        auto memory = test_engine.allocate_memory(key_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const int past_len = subsequence_descs[i].past_len;
            if (past_len == 0)
                continue;

            const int blocks_num = cldnn::ceil_div(past_len + 1, block_size);
            const int start_block_idx = block_indices[block_indices_begins[i]];

            for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                const int last_token_idx = (block_idx == blocks_num - 1) ? (past_len - block_size * block_idx) : block_size;

                for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                    for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                        const size_t input_token_offset = static_cast<size_t>(block_idx) * block_size + token_idx;
                        ov::float16* src_ptr =
                            key_data[i].data() + input_token_offset * static_cast<size_t>(num_kv_heads) * head_size + static_cast<size_t>(head_idx) * head_size;

                        if (!kv_cache_compression) {
                            const size_t base = (static_cast<size_t>(start_block_idx + block_idx) * num_kv_heads * block_size * head_size) +
                                                (static_cast<size_t>(head_idx) * block_size * head_size);
                            const size_t off = base + static_cast<size_t>(token_idx) * head_size;
                            set_values(test_stream, memory, src_ptr, head_size, off);
                        } else if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_TOKEN) {
                            // Compressed Key cache layout:
                            // logical shape: [num_blocks, num_kv_heads, block_size, adjusted_head_size], dt=i8 (adjusted_head_size=head_size+4).
                            // Per (block, head) region starts at block_base_i8, byte-packed as:
                            //   data:  block_base_i8 + t*head_size                 (u8 semantics), size=head_size bytes
                            //   scale: scale_base_i8 + t*sizeof(fp16)              (fp16), indexed as (scale_base_i8/2 + t)
                            //   zp:    zp_base_i8 + t*sizeof(fp16)                 (fp16), indexed as (zp_base_i8/2 + t)
                            // xattention quant: q∈[0..255], dequant x ≈ (q - zp) * scale, where scale=(max-min)/255, zp=(-min)*255/(max-min).
                            auto [qdata, scale, zp] = quantize_data(src_ptr, head_size, false, true);
                            int8_t* qptr = reinterpret_cast<int8_t*>(qdata.data());

                            const size_t block_stride_i8 = static_cast<size_t>(adjusted_head_size) * block_size;
                            const size_t block_base_i8 = (static_cast<size_t>(start_block_idx + block_idx) * num_kv_heads + head_idx) * block_stride_i8;

                            const size_t data_off_i8 = block_base_i8 + token_idx * head_size;
                            set_values(test_stream, memory, qptr, head_size, data_off_i8);

                            const size_t scale_base_i8 = block_base_i8 + head_size * block_size;
                            const size_t zp_base_i8 = scale_base_i8 + block_size * sizeof(ov::float16);

                            const size_t scale_off_f16 = scale_base_i8 / 2 + token_idx;
                            const size_t zp_off_f16 = zp_base_i8 / 2 + token_idx;

                            set_values(test_stream, memory, &scale, 1, scale_off_f16);
                            set_values(test_stream, memory, &zp, 1, zp_off_f16);
                        } else {
                            // Compressed Key cache layout for BY_CHANNEL:
                            // shape: [num_blocks, num_kv_heads, adjusted_block_size, head_size], dt=i8.
                            // Per (block, head):
                            //   data bytes region                            : [block_size * head_size]
                            //   scale fp16 region per-subblock per-channel   : [(block_size / sub_block) * head_size]
                            //   zp fp16 region per-subblock per-channel      : [(block_size / sub_block) * head_size]
                            const size_t block_stride_i8 = static_cast<size_t>(adjusted_block_size) * static_cast<size_t>(head_size);
                            const size_t block_base_i8 =
                                (static_cast<size_t>(start_block_idx + block_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(head_idx)) *
                                block_stride_i8;

                            const int subblock_count = block_size / kv_sub_block_size;
                            const size_t scale_base_i8 = block_base_i8 + static_cast<size_t>(block_size) * static_cast<size_t>(head_size);
                            const size_t zp_base_i8 =
                                scale_base_i8 + static_cast<size_t>(subblock_count) * static_cast<size_t>(head_size) * sizeof(ov::float16);

                            for (int channel = 0; channel < head_size; channel++) {
                                for (int sub_start = 0; sub_start < last_token_idx; sub_start += kv_sub_block_size) {
                                    const int cur_sub_block_size = std::min(kv_sub_block_size, last_token_idx - sub_start);
                                    std::vector<ov::float16> token_block(cur_sub_block_size);

                                    for (int t = 0; t < cur_sub_block_size; t++) {
                                        const size_t input_token_offset = static_cast<size_t>(block_idx) * block_size + static_cast<size_t>(sub_start + t);
                                        token_block[t] = *(key_data[i].data() + input_token_offset * static_cast<size_t>(num_kv_heads) * head_size +
                                                           static_cast<size_t>(head_idx) * head_size + channel);
                                    }

                                    auto [quantized_data, scale, zp] = quantize_data(token_block.data(), cur_sub_block_size, true, true);

                                    for (int t = 0; t < cur_sub_block_size; t++) {
                                        const size_t data_off_i8 = block_base_i8 + static_cast<size_t>(sub_start + t) * head_size + channel;
                                        set_values(test_stream, memory, quantized_data.data() + t, 1, data_off_i8);
                                    }

                                    const size_t sub_idx = static_cast<size_t>(sub_start / kv_sub_block_size);
                                    const size_t scale_off_f16 = scale_base_i8 / 2 + sub_idx * static_cast<size_t>(head_size) + channel;
                                    const size_t zp_off_f16 = zp_base_i8 / 2 + sub_idx * static_cast<size_t>(head_size) + channel;

                                    set_values(test_stream, memory, &scale, 1, scale_off_f16);
                                    set_values(test_stream, memory, &zp, 1, zp_off_f16);
                                }
                            }
                        }
                    }
                }
            }
        }
        return memory;
    }

    bool is_int4_kv_cache() const {
        return kv_cache_precision == ov::element::u4 || kv_cache_precision == ov::element::i4;
    }

    cldnn::memory::ptr get_key_cache_memory() {
        auto key_cache_dt = cldnn::data_types::f16;
        auto adjusted_head_size = k_head_size;
        auto adjusted_block_size = block_size;
        if (kv_cache_compression) {
            key_cache_dt = is_int4_kv_cache() ? cldnn::data_types::u8 : cldnn::data_types::i8;
            const int scale_zp_bytes = 4;  // 2 fp16 values (scale + zp) = 4 bytes
            if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                if (is_int4_kv_cache()) {
                    // u4/i4 BY_CHANNEL: block_size dim is packed (2 u4 tokens per byte) + scale/zp.
                    // Shape: [num_blocks, kv_heads, k_head_size, block_size/2 + 4]
                    // head_size is NOT packed (outer dim), block_size IS packed (inner dim).
                    adjusted_head_size = k_head_size;                       // NOT packed
                    adjusted_block_size = block_size / 2 + scale_zp_bytes;  // packed + scale/zp
                } else {
                    adjusted_block_size += scale_zp_bytes;
                }
            } else {
                if (is_int4_kv_cache()) {
                    // Scale/zp for BY_TOKEN: 2 fp16 values = 4 bytes appended to head_size dim.
                    adjusted_head_size = k_head_size / 2 + scale_zp_bytes;
                } else {
                    adjusted_head_size += scale_zp_bytes;
                }
            }
        }

        auto num_blocks = block_indices.back() + 1;
        auto key_cache_shape = ov::PartialShape{num_blocks, num_kv_heads, adjusted_head_size, adjusted_block_size};
        auto key_cache_layout = cldnn::layout{key_cache_shape, key_cache_dt, cldnn::format::bfyx};
        auto memory = test_engine.allocate_memory(key_cache_layout);
        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            int past_len = subsequence_descs[i].past_len;
            if (past_len != 0) {
                int blocks_num = cldnn::ceil_div(past_len + 1, block_size);
                int start_block_idx = block_indices[block_indices_begins[i]];
                for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                    int last_token_idx = block_idx == blocks_num - 1 ? (past_len - block_size * block_idx) : block_size;
                    // quantize by channel
                    if (kv_cache_compression && key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                        if (is_int4_kv_cache()) {
                            // INT4 BY_CHANNEL: packed layout [num_blocks, kv_heads, k_head_size, block_size/2+4]
                            // block_size dim is packed: 2 u4 tokens per byte along innermost dim.
                            // Comp region at [d, block_size/2..block_size/2+3]: 2 fp16 = inv_scale, zp per head dim d.
                            const int packed_block = block_size / 2;
                            for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                                for (int d = 0; d < k_head_size; d++) {
                                    // Gather values for this head dim across all tokens in this block
                                    std::vector<float> vals(block_size, 0.f);
                                    for (int t = 0; t < last_token_idx; ++t) {
                                        size_t in_off = (static_cast<size_t>(block_idx) * block_size + t) * num_kv_heads * k_head_size + head_idx * k_head_size;
                                        vals[t] = static_cast<float>(key_data[i].data()[in_off + d]);
                                    }
                                    // Quantize to u4
                                    float min_v = vals[0], max_v = vals[0];
                                    for (int t = 1; t < last_token_idx; ++t) {
                                        min_v = std::min(min_v, vals[t]);
                                        max_v = std::max(max_v, vals[t]);
                                    }
                                    float range = (max_v == min_v) ? 0.001f : (max_v - min_v);
                                    const float min_range = std::abs(max_v) * 0.1f;
                                    if (range <= min_range)
                                        range += std::max(1.0f, min_range);
                                    float scale = 15.0f / range;
                                    float zp_val = -min_v * scale;
                                    std::vector<uint8_t> q(last_token_idx);
                                    for (int t = 0; t < last_token_idx; ++t) {
                                        int v = static_cast<int>(std::nearbyint(vals[t] * scale + zp_val));
                                        q[t] = static_cast<uint8_t>(std::max(0, std::min(15, v)));
                                    }

                                    const size_t block_offset =
                                        static_cast<size_t>(start_block_idx + block_idx) * num_kv_heads * k_head_size * adjusted_block_size +
                                        head_idx * k_head_size * adjusted_block_size;
                                    const size_t row_offset = block_offset + d * adjusted_block_size;

                                    // Pack 2 u4 tokens per byte: token t0 in lower nibble, t1 in upper nibble
                                    std::vector<uint8_t> packed_data(packed_block, 0);
                                    for (int t = 0; t < last_token_idx; ++t) {
                                        int byte_idx = t / 2;
                                        if (t % 2 == 0)
                                            packed_data[byte_idx] = q[t] & 0xFu;
                                        else
                                            packed_data[byte_idx] |= (q[t] & 0xFu) << 4;
                                    }
                                    set_values(test_stream, memory, packed_data.data(), static_cast<size_t>(packed_block), row_offset);

                                    // Write comp: 2 fp16 (inv_scale, zp) at row_offset + packed_block
                                    const size_t comp_offset_fp16 = (row_offset + packed_block) / 2;
                                    ov::float16 inv_scale_val = static_cast<float>(1.0f / scale);
                                    ov::float16 fp16_zp = static_cast<float>(zp_val);
                                    set_values(test_stream, memory, &inv_scale_val, 1, comp_offset_fp16 + 0);
                                    set_values(test_stream, memory, &fp16_zp, 1, comp_offset_fp16 + 1);
                                }
                            }
                        } else {
                            for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                                for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                    std::vector<ov::float16> token_block(block_size);
                                    for (int token_idx = 0; token_idx < last_token_idx; ++token_idx) {
                                        size_t input_token_offset = block_idx * block_size + token_idx;
                                        token_block[token_idx] =
                                            *(key_data[i].data() + input_token_offset * num_kv_heads * k_head_size + head_idx * k_head_size + k_head_size_idx);
                                    }
                                    auto [quantized_data, scale, zp] = quantize_data(token_block.data(), last_token_idx, true);
                                    size_t output_block_offset = (start_block_idx + block_idx) * num_kv_heads * adjusted_head_size * adjusted_block_size +
                                                                 head_idx * adjusted_head_size * adjusted_block_size;
                                    size_t output_offset = output_block_offset + k_head_size_idx * adjusted_block_size;
                                    set_values(test_stream, memory, quantized_data.data(), last_token_idx, output_offset);
                                    size_t comp_offset = (output_offset + block_size) / 2;
                                    set_values(test_stream, memory, &scale, 1, comp_offset);
                                    set_values(test_stream, memory, &zp, 1, comp_offset + 1);
                                }
                            }
                        }
                    }
                    for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                        for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                            if (kv_cache_compression) {
                                if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_TOKEN) {
                                    // quantize by token
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    ov::float16* data_ptr = key_data[i].data() + input_token_offset * num_kv_heads * k_head_size + head_idx * k_head_size;
                                    // shape: [num_blocks, num_kv_heads, adjusted_head_size, block_size]
                                    size_t output_block_offset = (start_block_idx + block_idx) * num_kv_heads * adjusted_head_size * block_size +
                                                                 head_idx * adjusted_head_size * block_size;

                                    if (is_int4_kv_cache()) {
                                        // INT4 BY_TOKEN: [num_blocks, kv_heads, k_head_size/2+8, block_size] u8
                                        // Kernel packing (SUBGROUP_SIZE=16):
                                        //   Y = pack_group*16 + sglid, where pack_group = d/(2*16), sglid = d%16
                                        //   lower 4bit = q(dim[pack_group*32+sglid])
                                        //   upper 4bit = q(dim[pack_group*32+sglid+16])
                                        // Scale/ZP: fp16 in comp region at Y=packed_head_size..
                                        //   inv_scale[t] at fp16 idx (comp_base/2 + t)
                                        //   zp[t]        at fp16 idx (comp_base/2 + block_size + t)
                                        const int packed_head_size = k_head_size / 2;
                                        constexpr int SG = 16;  // SUBGROUP_SIZE
                                        // Compute per-token min/max, then scale/zp in u4 range [0,15]
                                        float min_v = std::numeric_limits<float>::max();
                                        float max_v = -std::numeric_limits<float>::max();
                                        for (int d = 0; d < k_head_size; d++) {
                                            float v = static_cast<float>(data_ptr[d]);
                                            min_v = std::min(min_v, v);
                                            max_v = std::max(max_v, v);
                                        }
                                        float range = (max_v == min_v) ? 0.001f : (max_v - min_v);
                                        const float min_range = std::abs(max_v) * 0.1f;
                                        if (range <= min_range)
                                            range += std::max(1.0f, min_range);
                                        float token_scale = 15.0f / range;
                                        float token_zp = -min_v * token_scale;
                                        std::vector<uint8_t> q(k_head_size);
                                        for (int d = 0; d < k_head_size; d++) {
                                            int v = static_cast<int>(std::nearbyint(static_cast<float>(data_ptr[d]) * token_scale + token_zp));
                                            q[d] = static_cast<uint8_t>(std::max(0, std::min(15, v)));
                                        }
                                        // Pack and write: Y=pack_group*SG+sglid → (lower=q[d0], upper=q[d1])
                                        for (int y = 0; y < packed_head_size; y++) {
                                            int sglid_val = y % SG;
                                            int pack_group = y / SG;
                                            int d0 = pack_group * 2 * SG + sglid_val;
                                            int d1 = d0 + SG;
                                            uint8_t packed_byte = q[d0] & 0xFu;
                                            if (d1 < k_head_size)
                                                packed_byte |= (q[d1] & 0xFu) << 4;
                                            size_t offset = output_block_offset + static_cast<size_t>(y) * block_size + token_idx;
                                            set_values(test_stream, memory, &packed_byte, 1, offset);
                                        }
                                        // Write inv_scale and zp as fp16 in the comp region
                                        size_t comp_offset_fp16 = (output_block_offset + static_cast<size_t>(packed_head_size) * block_size) / 2;
                                        ov::float16 fp16_inv_scale = static_cast<float>(1.0f / token_scale);
                                        ov::float16 fp16_zp = static_cast<float>(token_zp);
                                        set_values(test_stream, memory, &fp16_inv_scale, 1, comp_offset_fp16 + token_idx);
                                        set_values(test_stream, memory, &fp16_zp, 1, comp_offset_fp16 + block_size + token_idx);
                                    } else {
                                        auto [quantized_data, scale, zp] = quantize_data(data_ptr, k_head_size);
                                        for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                            auto quantized_data_ptr = quantized_data.data() + k_head_size_idx;

                                            size_t output_offset = output_block_offset + k_head_size_idx * block_size + token_idx;

                                            set_values(test_stream, memory, quantized_data_ptr, 1, output_offset);
                                        }
                                        size_t comp_offset = (output_block_offset + k_head_size * block_size) / 2;
                                        set_values(test_stream, memory, &scale, 1, comp_offset + token_idx);
                                        set_values(test_stream, memory, &zp, 1, comp_offset + block_size + token_idx);
                                    }
                                }
                            } else {
                                for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    ov::float16* data_ptr =
                                        key_data[i].data() + input_token_offset * num_kv_heads * k_head_size + head_idx * k_head_size + k_head_size_idx;

                                    // shape: [num_blocks, num_kv_heads, k_head_size, block_size]
                                    size_t output_offset = (start_block_idx + block_idx) * num_kv_heads * k_head_size * block_size +
                                                           head_idx * k_head_size * block_size + k_head_size_idx * block_size + token_idx;

                                    set_values(test_stream, memory, data_ptr, 1, output_offset);
                                }
                            }
                        }
                    }
                }
            }
        }

        return memory;
    }

    cldnn::memory::ptr get_value_cache_memory() {
        auto value_cache_dt = cldnn::data_types::f16;
        const int head_size = v_head_size;
        int scale_zp_bytes = 0;
        if (kv_cache_compression) {
            value_cache_dt = is_int4_kv_cache() ? cldnn::data_types::u8 : cldnn::data_types::i8;
            scale_zp_bytes = 4;  // 2 fp16 values (scale + zp) = 4 bytes
        }

        // For u4 (INT4), values are packed 2 per byte; the physical head size is halved.
        // PACKED_ADJUSTED_V_HEAD_SIZE = v_head_size/2 + scales_zp_size = 32 + 4 = 36.
        const int adjusted_head_size = is_int4_kv_cache() ? (head_size / 2 + scale_zp_bytes) : (head_size + scale_zp_bytes);

        const auto num_blocks = block_indices.back() + 1;
        auto value_cache_shape = ov::PartialShape{static_cast<int64_t>(num_blocks),
                                                  static_cast<int64_t>(num_kv_heads),
                                                  static_cast<int64_t>(block_size),
                                                  static_cast<int64_t>(adjusted_head_size)};
        auto value_cache_layout = cldnn::layout{value_cache_shape, value_cache_dt, cldnn::format::bfyx};
        auto memory = test_engine.allocate_memory(value_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const int past_len = subsequence_descs[i].past_len;
            if (past_len == 0)
                continue;

            const int blocks_num = cldnn::ceil_div(past_len + 1, block_size);
            const int start_block_idx = block_indices[block_indices_begins[i]];

            for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                const int last_token_idx = (block_idx == blocks_num - 1) ? (past_len - block_size * block_idx) : block_size;

                for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                    for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                        const size_t input_token_offset = static_cast<size_t>(block_idx) * block_size + token_idx;

                        ov::float16* src_ptr = value_data[i].data() + input_token_offset * static_cast<size_t>(num_kv_heads) * head_size +
                                               static_cast<size_t>(head_idx) * head_size;

                        if (!kv_cache_compression) {
                            const size_t base = (static_cast<size_t>(start_block_idx + block_idx) * static_cast<size_t>(num_kv_heads) *
                                                 static_cast<size_t>(block_size) * static_cast<size_t>(head_size)) +
                                                (static_cast<size_t>(head_idx) * static_cast<size_t>(block_size) * static_cast<size_t>(head_size));
                            const size_t off = base + static_cast<size_t>(token_idx) * static_cast<size_t>(head_size);
                            set_values(test_stream, memory, src_ptr, head_size, off);
                        } else if (is_int4_kv_cache()) {
                            // INT4 (u4) BY_TOKEN value cache: inline per-token comp layout.
                            // [num_blocks, kv_heads, block_size, PACKED_ADJUSTED_V_HEAD_SIZE] u8
                            // PACKED_ADJUSTED_V_HEAD_SIZE = v_head_size/2 + 4 = 36 (for v_head_size=64).
                            // Per token: [packed_data (32 bytes) | scale (fp16) | zp (fp16)] = 36 bytes.
                            const int packed_head_size = head_size / 2;

                            // Quantize entire token: one scale/zp for all head dims (BY_TOKEN)
                            float min_val = std::numeric_limits<float>::max();
                            float max_val = std::numeric_limits<float>::lowest();
                            for (int d = 0; d < head_size; d++) {
                                float v = static_cast<float>(src_ptr[d]);
                                min_val = std::min(min_val, v);
                                max_val = std::max(max_val, v);
                            }
                            float diff = (max_val == min_val) ? 0.001f : (max_val - min_val);
                            float min_range = std::abs(max_val * 0.1f);
                            if (diff <= min_range)
                                diff += std::max(1.0f, min_range);
                            float scale_val = 15.0f / diff;
                            float zp_val = -min_val * scale_val;
                            ov::float16 inv_scale_fp16 = ov::float16(1.0f / scale_val);
                            ov::float16 zp_fp16 = ov::float16(zp_val);

                            const size_t block_stride = static_cast<size_t>(adjusted_head_size) * static_cast<size_t>(block_size);
                            const size_t block_base =
                                (static_cast<size_t>(start_block_idx + block_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(head_idx)) *
                                block_stride;

                            // Token base: each token occupies adjusted_head_size bytes (inline comp)
                            const size_t token_base = block_base + static_cast<size_t>(token_idx) * static_cast<size_t>(adjusted_head_size);

                            // Pack pairs of groups: group (g*2) and (g*2+1), each 16 dims wide.
                            const int num_packed_groups = packed_head_size / 16;
                            for (int g = 0; g < num_packed_groups; g++) {
                                for (int lane = 0; lane < 16; lane++) {
                                    int dim_even = (g * 2) * 16 + lane;
                                    int dim_odd = (g * 2 + 1) * 16 + lane;
                                    int q0 = static_cast<int>(std::nearbyint(static_cast<float>(src_ptr[dim_even]) * scale_val + zp_val));
                                    int q1 = static_cast<int>(std::nearbyint(static_cast<float>(src_ptr[dim_odd]) * scale_val + zp_val));
                                    q0 = std::max(0, std::min(15, q0));
                                    q1 = std::max(0, std::min(15, q1));
                                    uint8_t packed_byte = static_cast<uint8_t>((q0 & 0xFu) | (static_cast<uint8_t>(q1 & 0xFu) << 4));
                                    const size_t packed_pos = g * 16 + lane;
                                    set_values(test_stream, memory, &packed_byte, 1, token_base + packed_pos);
                                }
                            }

                            // Write inline comp: scale (fp16) and zp (fp16) right after packed data
                            const size_t comp_byte_off = token_base + static_cast<size_t>(packed_head_size);
                            const size_t comp_f16_off = comp_byte_off / 2;
                            set_values(test_stream, memory, &inv_scale_fp16, 1, comp_f16_off);
                            set_values(test_stream, memory, &zp_fp16, 1, comp_f16_off + 1);
                        } else {
                            // Compressed Value cache layout:
                            // logical shape: [num_blocks, num_kv_heads, block_size, adjusted_head_size], dt=i8 (adjusted_head_size=head_size+4).
                            // Per (block, head): data at block_base_i8 + t*head_size; scale/zp are fp16 arrays at scale_base_i8/zp_base_i8
                            // (fp16 element offsets: scale_base_i8/2 + t, zp_base_i8/2 + t).
                            // has_xattention uses unsigned [0..255] quant; dequant x ≈ (q - zp) * scale, scale=(max-min)/255, zp=(-min)*255/(max-min).
                            auto [qdata, scale, zp] = quantize_data(src_ptr, head_size, false, has_xattention);
                            int8_t* qptr = reinterpret_cast<int8_t*>(qdata.data());

                            const size_t block_stride_i8 = static_cast<size_t>(adjusted_head_size) * static_cast<size_t>(block_size);
                            const size_t block_base_i8 =
                                (static_cast<size_t>(start_block_idx + block_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(head_idx)) *
                                block_stride_i8;

                            const size_t data_off_i8 = block_base_i8 + static_cast<size_t>(token_idx) * static_cast<size_t>(head_size);
                            set_values(test_stream, memory, qptr, head_size, data_off_i8);

                            const size_t scale_base_i8 = block_base_i8 + static_cast<size_t>(head_size) * static_cast<size_t>(block_size);
                            const size_t zp_base_i8 = scale_base_i8 + static_cast<size_t>(block_size) * sizeof(ov::float16);

                            const size_t scale_off_f16 = (scale_base_i8 >> 1) + static_cast<size_t>(token_idx);
                            const size_t zp_off_f16 = (zp_base_i8 >> 1) + static_cast<size_t>(token_idx);

                            set_values(test_stream, memory, &scale, 1, scale_off_f16);
                            set_values(test_stream, memory, &zp, 1, zp_off_f16);
                        }
                    }
                }
            }
        }
        return memory;
    }

    cldnn::memory::ptr get_past_lens_memory() {
        return get_memory_from_vec(past_lens);
    }

    cldnn::memory::ptr get_subsequence_begins_memory() {
        return get_memory_from_vec(subsequence_begins);
    }

    cldnn::memory::ptr get_block_indices_memory() {
        return get_memory_from_vec(block_indices);
    }

    cldnn::memory::ptr get_block_indices_begins_memory() {
        return get_memory_from_vec(block_indices_begins);
    }

    cldnn::memory::ptr get_scale_memory() {
        std::vector<ov::float16> scale = {ov::float16(get_default_scale())};
        return get_memory_from_vec(scale);
    }

    cldnn::memory::ptr get_sliding_window_memory() {
        std::vector<int> sliding_window = {0};
        return get_memory_from_vec(sliding_window);
    }

    cldnn::memory::ptr get_alibi_memory() {
        std::vector<ov::float16> alibi;
        return get_memory_from_vec(alibi);
    }

    cldnn::memory::ptr get_max_context_len_memory() {
        return get_memory_from_vec(max_context_len);
    }

    cldnn::memory::ptr get_score_aggregation() {
        return get_memory_from_vec(score_aggregation);
    }

    cldnn::memory::ptr get_rotated_block_indices_memory() {
        return get_memory_from_vec(rotated_block_indices);
    }

    cldnn::memory::ptr get_rotation_deltas_memory() {
        auto mem = get_memory_from_vec(rotation_deltas);
        auto layout = mem->get_layout();
        auto last_dim = rotation_config.per_block ? 1 : block_size;
        layout.set_partial_shape(ov::PartialShape{static_cast<long int>(rotated_block_indices.size()), last_dim});

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    cldnn::memory::ptr get_rotation_trig_lut_memory() {
        auto mem = get_memory_from_vec(rotation_trig_lut);
        auto layout = mem->get_layout();
        layout.set_partial_shape(ov::PartialShape{max_context_len[0], k_head_size});

        if (rotated_block_indices.empty()) {
            auto empty_layout = mem->get_layout();
            empty_layout.set_partial_shape(ov::PartialShape{0, k_head_size});
            return test_engine.reinterpret_buffer(*mem, empty_layout);
        }

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    cldnn::memory::ptr get_xattention_threshold_memory() {
        return get_memory_from_vec(xattention_threshold);
    }

    cldnn::memory::ptr get_xattention_block_size_memory() {
        return get_memory_from_vec(xattention_block_size);
    }

    cldnn::memory::ptr get_xattention_stride_memory() {
        return get_memory_from_vec(xattention_stride);
    }

    cldnn::memory::ptr get_sinks_memory() {
        auto mem = get_memory_from_vec(sinks);
        auto layout = mem->get_layout();
        layout.set_partial_shape(ov::PartialShape{1, num_heads, 1, 1});

        if (sinks.empty()) {
            auto empty_layout = mem->get_layout();
            empty_layout.set_partial_shape(ov::PartialShape{0, 0, 0, 0});
            return test_engine.reinterpret_buffer(*mem, empty_layout);
        }

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    cldnn::memory::ptr get_adaptive_rkv_start_size_memory() {
        auto mem = test_engine.allocate_memory({{}, cldnn::data_types::i32, cldnn::format::bfyx});
        cldnn::mem_lock<int> lock(mem, test_stream);
        lock[0] = adaptive_rkv_start_size;
        return mem;
    }

    cldnn::memory::ptr get_adaptive_rkv_evictable_sizes_memory() {
        return get_memory_from_vec(adaptive_rkv_evictable_sizes);
    }

    cldnn::memory::ptr get_adaptive_rkv_diversity_block_set_indices_memory() {
        return get_memory_from_vec(adaptive_rkv_diversity_block_set_indices);
    }

    cldnn::memory::ptr get_adaptive_rkv_diversity_block_set_indices_begins_memory() {
        return get_memory_from_vec(adaptive_rkv_diversity_block_set_indices_begins);
    }

    cldnn::memory::ptr get_token_type_ids_memory() {
        if (!token_type_ids.empty()) {
            return get_memory_from_vec(token_type_ids);
        }
        std::vector<int> default_token_type_ids = {0};
        return get_memory_from_vec(default_token_type_ids);
    }

    cldnn::memory::ptr get_qq_bias_memory() {
        std::vector<uint8_t> flat_qq_bias;
        for (const auto& matrix : qq_bias) {
            for (bool val : matrix) {
                flat_qq_bias.push_back(static_cast<int8_t>(val));
            }
        }
        return get_memory_from_vec(flat_qq_bias);
    }

    cldnn::memory::ptr get_qq_bias_begins_memory() {
        return get_memory_from_vec(qq_bias_begins);
    }

    float get_default_scale() {
        return static_cast<float>(1.f / std::sqrt(k_head_size));
    }

private:
    template <typename T>
    cldnn::memory::ptr get_memory_from_vec(std::vector<T>& input_data) {
        auto data_size = input_data.empty() ? 1 : input_data.size();
        auto shape = ov::PartialShape{static_cast<int>(data_size)};
        auto layout = cldnn::layout{shape, ov::element::from<T>(), cldnn::format::bfyx};
        auto memory = test_engine.allocate_memory(layout);

        if (input_data.empty()) {
            auto shape = ov::PartialShape{0};
            auto layout = cldnn::layout{shape, ov::element::from<T>(), cldnn::format::bfyx};
            return test_engine.reinterpret_buffer(*memory, layout);
        }

        set_values(test_stream, memory, input_data.data(), input_data.size(), 0);

        return memory;
    }

    cldnn::memory::ptr get_QKV_memory(std::vector<std::vector<ov::float16>>& input_data, int num_heads, int head_size, bool skip_past_len) {
        int total_tokens = 0;
        for (const auto& subsequence_desc : subsequence_descs)
            total_tokens += subsequence_desc.num_tokens;

        auto query_shape = ov::PartialShape{total_tokens, num_heads * head_size};
        auto query_layout = cldnn::layout{query_shape, cldnn::data_types::f16, cldnn::format::bfyx};
        auto memory = test_engine.allocate_memory(query_layout);

        for (int subsequence_idx = 0; subsequence_idx < static_cast<int>(subsequence_descs.size()); subsequence_idx++) {
            for (int token_idx = 0; token_idx < subsequence_descs[subsequence_idx].num_tokens; token_idx++) {
                for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                    size_t input_token_offset = token_idx;
                    // as generated data stored in vectors includes past_len, ignore it for KV inputs
                    if (skip_past_len)
                        input_token_offset += subsequence_descs[subsequence_idx].past_len;

                    ov::float16* data_ptr = input_data[subsequence_idx].data() + input_token_offset * num_heads * head_size + head_idx * head_size;

                    size_t output_token_offset = subsequence_begins[subsequence_idx] + token_idx;
                    size_t output_offset = output_token_offset * num_heads * head_size + head_idx * head_size;

                    set_values(test_stream, memory, data_ptr, head_size, output_offset);
                }
            }
        }

        return memory;
    }

    template <typename T>
    static void set_values(cldnn::stream& stream, cldnn::memory::ptr mem, T* vals, size_t size, size_t dst_offset) {
        cldnn::mem_lock<T> mem_ptr(mem, stream);
        for (size_t i = 0; i < size; i++) {
            mem_ptr[dst_offset + i] = vals[i];
        }
    }

    static std::vector<ov::float16> generate_input_data(tests::random_generator& rg, size_t num_heads, size_t tokens_num, size_t k_head_size) {
        const size_t total_elements_num = tokens_num * num_heads * k_head_size;
        auto data = rg.generate_random_1d<ov::float16>(total_elements_num, -1, 1);

        return data;
    }

    static std::vector<ov::float16> generate_realistic_data(size_t num_heads, size_t tokens_num, size_t k_head_size) {
        std::vector<ov::float16> data(num_heads * tokens_num * k_head_size);

        std::mt19937 gen(1234);
        std::normal_distribution<float> dist(0.0f, 0.1f);

        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t t = 0; t < tokens_num; ++t) {
                for (size_t d = 0; d < k_head_size; ++d) {
                    float val = dist(gen);
                    if (t > 0)
                        val = 0.8f * val + 0.2f * static_cast<float>(data[h * tokens_num * k_head_size + (t - 1) * k_head_size + d]);
                    data[h * tokens_num * k_head_size + t * k_head_size + d] = static_cast<ov::float16>(val);
                }
            }
        }

        return data;
    }

    static std::vector<int> generate_rotation_deltas_data(tests::random_generator& rg,
                                                          size_t max_tokens_num,
                                                          size_t rotated_blocks_num,
                                                          size_t block_size,
                                                          bool per_block) {
        const size_t total_elements_num = per_block ? rotated_blocks_num : rotated_blocks_num * block_size;
        auto data = rg.generate_random_1d<int>(total_elements_num, 0, static_cast<int>(max_tokens_num - 1));

        return data;
    }

    static std::vector<ov::float16> generate_rotation_trig_lut_data(tests::random_generator& rg, size_t max_tokens_num, size_t k_head_size) {
        const size_t total_elements_num = max_tokens_num * k_head_size;
        auto data = rg.generate_random_1d<ov::float16>(total_elements_num, -1, 1);

        return data;
    }

    static std::tuple<std::vector<int8_t>, ov::float16, ov::float16> quantize_data(ov::float16* data,
                                                                                   size_t size,
                                                                                   bool expand_range = false,
                                                                                   bool has_xattention = false) {
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < size; i++) {
            float v = static_cast<float>(data[i]);
            min_value = std::min(min_value, v);
            max_value = std::max(max_value, v);
        }

        if (has_xattention) {
            if (max_value == min_value) {
                std::vector<int8_t> qdata(size, 0);
                return {qdata, ov::float16(0.0f), ov::float16(min_value)};
            }

            float diff_value = max_value - min_value;
            if (expand_range && std::abs(diff_value) <= std::abs(max_value) * 0.1f) {
                diff_value = (max_value - min_value) + std::max(1.0f, max_value * 0.1f);
            }

            float scale_val = 255.0f / diff_value;
            float zp_val = -min_value * scale_val;

            std::vector<int8_t> qdata(size);
            for (size_t i = 0; i < size; i++) {
                float q = data[i] * scale_val + zp_val;
                int v = static_cast<int>(std::nearbyint(q));
                if (v < 0)
                    v = 0;
                if (v > 255)
                    v = 255;
                qdata[i] = static_cast<int8_t>(v);
            }

            ov::float16 scale = static_cast<float>(diff_value / 255.0f);
            ov::float16 zp = static_cast<float>(zp_val);
            return {qdata, scale, zp};
        }

        float diff_value = 0.001f;
        if (max_value != min_value)
            diff_value = max_value - min_value;
        if (expand_range && std::abs(diff_value) <= std::abs(max_value) * 0.1f) {
            diff_value = (max_value - min_value) + std::max(1.0f, max_value * 0.1f);
        }

        float scale = (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::lowest()) / diff_value;
        float zp = -min_value * scale + std::numeric_limits<int8_t>::lowest();

        std::vector<int8_t> qdata(size);
        auto convert_char_rte = [](float val) {
            float rounded = std::nearbyint(val);
            if (rounded > 127.0f)
                return static_cast<int8_t>(127);
            if (rounded < -128.0f)
                return static_cast<int8_t>(-128);
            return static_cast<int8_t>(rounded);
        };

        for (size_t i = 0; i < size; i++) {
            qdata[i] = convert_char_rte(data[i] * scale + zp);
        }

        ov::float16 scale_out = static_cast<float>(1.0f / scale);
        ov::float16 zp_out = static_cast<float>(zp);
        return {qdata, scale_out, zp_out};
    }
};

namespace std {
template <>
struct hash<ov::float16> {
    size_t operator()(const ov::float16& value) const noexcept {
        return std::hash<float>{}(static_cast<float>(value));
    }
};
}  // namespace std

struct PagedAttentionReference {
    PagedAttentionReference(PagedAttentionManager& pam) : pam(pam), test_engine(pam.test_engine), test_stream(pam.test_stream) {}

    std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> get_reference(cldnn::memory::ptr key_cache_mem = nullptr) {
        const bool has_xattention = pam.has_xattention;
        if (has_xattention) {
            const size_t total_iterations = pam.subsequence_descs.size();
            if (pam.xattention_threshold.size() != total_iterations) {
                OPENVINO_THROW("xattention_threshold size (", pam.xattention_threshold.size(), ") must match number of subsequences (", total_iterations, ")");
            }
            if (pam.xattention_block_size.size() != total_iterations) {
                OPENVINO_THROW("xattention_block_size size (",
                               pam.xattention_block_size.size(),
                               ") must match number of subsequences (",
                               total_iterations,
                               ")");
            }
        }

        std::vector<ov::float16> ref_data_output;
        std::vector<ov::float16> ref_scores_output;
        std::vector<ov::float16> ref_diversity_output;
        size_t qq_bias_offset = 0;
        for (size_t i = 0; i < pam.subsequence_descs.size(); i++) {
            const auto& subsequence_desc = pam.subsequence_descs[i];
            const auto kv_seq_len = subsequence_desc.num_tokens + subsequence_desc.past_len;

            auto key_data = pam.key_data[i];
            if (pam.rotation_config.apply_rotation) {
                auto blocks_start = pam.block_indices_begins[i];
                auto blocks_end = pam.block_indices_begins[i + 1];

                std::vector<int> block_indices(pam.block_indices.begin() + blocks_start, pam.block_indices.begin() + blocks_end);

                for (const auto& block_idx : block_indices) {
                    auto it = std::find(pam.rotated_block_indices.begin(), pam.rotated_block_indices.end(), block_idx);
                    if (it != pam.rotated_block_indices.end()) {
                        int index = std::distance(pam.rotated_block_indices.begin(), it);
                        int subsequence_rotated_block_idx = *it - blocks_start;

                        rotate_block(key_data,
                                     pam.rotation_deltas,
                                     pam.rotation_trig_lut,
                                     index,
                                     subsequence_rotated_block_idx,
                                     pam.num_kv_heads,
                                     pam.k_head_size,
                                     pam.block_size,
                                     pam.rotation_config.per_block);
                    }
                }
            }

            auto window_size = pam.has_score_aggregation ? pam.score_aggregation[i] : 1;

            const std::vector<uint8_t>* qq_bias_ptr = nullptr;
            if (pam.qq_bias.size() > 0 && pam.subsequence_descs[i].past_len != 0) {
                qq_bias_ptr = &pam.qq_bias[qq_bias_offset++];
            }

            double xattn_threshold = 1.0;
            size_t xattn_block_size = 128;
            if (has_xattention) {
                xattn_threshold = static_cast<double>(pam.xattention_threshold[i]);

                //  reference path reflects runtime fallback/validation behavior for block size.
                if (test_engine.get_device_info().arch < cldnn::gpu_arch::xe2) {
                    xattn_block_size = 128;
                } else {
                    const int user_value = pam.xattention_block_size[i];
                    xattn_block_size = (user_value == 128 || user_value == 256) ? static_cast<size_t>(user_value) : 256;
                }
            }

            auto subsequence_ref_results = run_reference(has_xattention,
                                                         pam.query_data[i],
                                                         key_data,
                                                         pam.value_data[i],
                                                         subsequence_desc.num_tokens,
                                                         kv_seq_len,
                                                         pam.num_heads,
                                                         pam.num_kv_heads,
                                                         pam.k_head_size,
                                                         pam.v_head_size,
                                                         window_size,
                                                         pam.sliding_window_size,
                                                         pam.get_default_scale(),
                                                         xattn_threshold,
                                                         xattn_block_size,
                                                         qq_bias_ptr);

            // concatenate all subsequences into one vector
            ref_data_output.insert(ref_data_output.end(), subsequence_ref_results.first.begin(), subsequence_ref_results.first.end());
            ref_scores_output.insert(ref_scores_output.end(), subsequence_ref_results.second.begin(), subsequence_ref_results.second.end());
        }

        if (!pam.adaptive_rkv_evictable_sizes.empty()) {
            ref_diversity_output = compute_diversity_reference(key_cache_mem);
        }

        return {ref_data_output, ref_scores_output, ref_diversity_output};
    }

private:
    std::pair<std::vector<ov::float16>, std::vector<ov::float16>> run_reference(bool has_xattention,
                                                                                const std::vector<ov::float16>& query_data,
                                                                                const std::vector<ov::float16>& key_data,
                                                                                const std::vector<ov::float16>& value_data,
                                                                                int num_queries,
                                                                                int num_keys,
                                                                                int num_heads,
                                                                                int num_kv_heads,
                                                                                int k_head_size,
                                                                                int v_head_size,
                                                                                int window_size,
                                                                                int sliding_window_size,
                                                                                float scale,
                                                                                double xattention_threshold,
                                                                                size_t block_size,
                                                                                const std::vector<uint8_t>* qq_bias = nullptr,
                                                                                size_t stride = 16) {
        auto query_shape = ov::PartialShape{1, num_queries, num_heads, k_head_size};
        auto key_shape = ov::PartialShape{1, num_keys, num_kv_heads, k_head_size};
        auto value_shape = ov::PartialShape{1, num_keys, num_kv_heads, v_head_size};
        if (num_heads != num_kv_heads && !has_xattention) {
            query_shape = ov::PartialShape{num_queries, num_kv_heads, (num_heads / num_kv_heads), k_head_size};
            key_shape = ov::PartialShape{num_keys, num_kv_heads, 1, k_head_size};
            value_shape = ov::PartialShape{num_keys, num_kv_heads, 1, v_head_size};
        }
        bool do_gqa_expand = false;
        std::vector<ov::float16> expanded_key_data;
        std::vector<ov::float16> expanded_value_data;
        if (has_xattention) {
            // Grouped Query Attention
            do_gqa_expand = (num_heads != num_kv_heads);
            if (do_gqa_expand) {
                const int group_size = num_heads / num_kv_heads;

                expanded_key_data.resize(static_cast<size_t>(num_keys) * static_cast<size_t>(num_heads) * static_cast<size_t>(k_head_size));
                expanded_value_data.resize(static_cast<size_t>(num_keys) * static_cast<size_t>(num_heads) * static_cast<size_t>(v_head_size));

                for (int key_idx = 0; key_idx < num_keys; ++key_idx) {
                    for (int h = 0; h < num_heads; ++h) {
                        const int src_kv_head = h / group_size;
                        size_t src_key_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(src_kv_head)) *
                                             static_cast<size_t>(k_head_size);
                        size_t dst_key_off =
                            (static_cast<size_t>(key_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) * static_cast<size_t>(k_head_size);
                        for (int d = 0; d < k_head_size; ++d)
                            expanded_key_data[dst_key_off + static_cast<size_t>(d)] = key_data[src_key_off + static_cast<size_t>(d)];

                        size_t src_val_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(src_kv_head)) *
                                             static_cast<size_t>(v_head_size);
                        size_t dst_val_off =
                            (static_cast<size_t>(key_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) * static_cast<size_t>(v_head_size);
                        for (int d = 0; d < v_head_size; ++d)
                            expanded_value_data[dst_val_off + static_cast<size_t>(d)] = value_data[src_val_off + static_cast<size_t>(d)];
                    }
                }

                key_shape = ov::PartialShape{1, num_keys, num_heads, k_head_size};
                value_shape = ov::PartialShape{1, num_keys, num_heads, v_head_size};
                num_kv_heads = num_heads;
            }
        }

        auto query_layout = cldnn::layout{query_shape, cldnn::data_types::f16, cldnn::format::bfyx};
        auto key_layout = cldnn::layout{key_shape, cldnn::data_types::f16, cldnn::format::bfyx};
        auto value_layout = cldnn::layout{value_shape, cldnn::data_types::f16, cldnn::format::bfyx};
        auto scale_layout = cldnn::layout({1}, cldnn::data_types::f16, cldnn::format::bfyx);

        OPENVINO_ASSERT(query_layout.count() == query_data.size());
        if (do_gqa_expand) {
            OPENVINO_ASSERT(key_layout.count() == expanded_key_data.size());
            OPENVINO_ASSERT(value_layout.count() == expanded_value_data.size());
        } else {
            OPENVINO_ASSERT(key_layout.count() == key_data.size());
            OPENVINO_ASSERT(value_layout.count() == value_data.size());
        }

        auto query_mem = test_engine.allocate_memory(query_layout);
        auto key_mem = test_engine.allocate_memory(key_layout);
        auto value_mem = test_engine.allocate_memory(value_layout);
        auto scale_mem = test_engine.allocate_memory(scale_layout);

        tests::set_values(query_mem, query_data);
        if (do_gqa_expand) {
            tests::set_values(key_mem, expanded_key_data);
            tests::set_values(value_mem, expanded_value_data);
        } else {
            tests::set_values(key_mem, key_data);
            tests::set_values(value_mem, value_data);
        }
        tests::set_values(scale_mem, {static_cast<ov::float16>(scale)});

        ov::reference::XAttentionRetainedBlockIndicesForAllHeads retained_blocks;
        if (num_queries >= static_cast<int>(block_size) && has_xattention) {
            auto reorder_qhk_to_hqd = [&](const std::vector<ov::float16>& src, int outer_len, int num_heads, int head_dim) {
                std::vector<ov::float16> dst(num_heads * outer_len * head_dim);
                for (int h = 0; h < num_heads; ++h) {
                    size_t dst_h_off = static_cast<size_t>(h) * outer_len * head_dim;
                    for (int i = 0; i < outer_len; ++i) {
                        size_t src_off = static_cast<size_t>(i) * num_heads * head_dim + static_cast<size_t>(h) * head_dim;
                        std::copy_n(&src[src_off], head_dim, &dst[dst_h_off + static_cast<size_t>(i) * head_dim]);
                    }
                }
                return dst;
            };

            const auto query_data_3d = reorder_qhk_to_hqd(query_data, num_queries, num_heads, k_head_size);
            const auto key_data_3d = reorder_qhk_to_hqd(do_gqa_expand ? expanded_key_data : key_data, num_keys, num_heads, k_head_size);
            const size_t padded_q = ((num_queries + block_size - 1) / block_size) * block_size;
            const size_t padded_k = ((num_keys + block_size - 1) / block_size) * block_size;

            std::vector<float> query_padded(num_heads * padded_q * k_head_size, 0.f);
            std::vector<float> key_padded(num_heads * padded_k * k_head_size, 0.f);

            for (int h = 0; h < num_heads; ++h) {
                const auto* q_src = &query_data_3d[h * num_queries * k_head_size];
                const auto* k_src = &key_data_3d[h * num_keys * k_head_size];
                auto* q_dst = &query_padded[h * padded_q * k_head_size];
                auto* k_dst = &key_padded[h * padded_k * k_head_size];

                std::transform(q_src, q_src + num_queries * k_head_size, q_dst, [](ov::float16 v) {
                    return static_cast<float>(v);
                });
                std::transform(k_src, k_src + num_keys * k_head_size, k_dst, [](ov::float16 v) {
                    return static_cast<float>(v);
                });
            }
            ov::reference::XAttentionBlockSelector<float> selector(xattention_threshold, block_size, stride);
            retained_blocks = selector.select_blocks(query_padded.data(),
                                                     {static_cast<size_t>(num_heads), padded_q, static_cast<size_t>(k_head_size)},
                                                     key_padded.data(),
                                                     {static_cast<size_t>(num_heads), padded_k, static_cast<size_t>(k_head_size)});
        }
        auto mask_mem = get_mask_mem_combined_multi_head(num_queries,
                                                         num_keys,
                                                         num_heads,
                                                         num_kv_heads,
                                                         sliding_window_size,
                                                         retained_blocks,
                                                         static_cast<int>(block_size),
                                                         qq_bias);
        cldnn::topology topology;
        if (num_heads == num_kv_heads) {
            topology.add(
                cldnn::input_layout("query", query_layout),
                cldnn::input_layout("key", key_layout),
                cldnn::input_layout("value", value_layout),
                cldnn::data("mask", mask_mem),
                cldnn::data("scale", scale_mem),
                cldnn::permute("query_transposed", cldnn::input_info("query"), {0, 2, 1, 3}),
                cldnn::permute("key_transposed", cldnn::input_info("key"), {0, 2, 3, 1}),
                cldnn::permute("value_transposed", cldnn::input_info("value"), {0, 2, 1, 3}),
                cldnn::gemm("qk_gemm", {cldnn::input_info("query_transposed"), cldnn::input_info("key_transposed")}, cldnn::data_types::f16, false, false),
                cldnn::eltwise("scale_div", {cldnn::input_info("qk_gemm"), cldnn::input_info("scale")}, cldnn::eltwise_mode::prod),
                cldnn::eltwise("eltwise", {cldnn::input_info("scale_div"), cldnn::input_info("mask")}, cldnn::eltwise_mode::sum),
                cldnn::softmax("softmax", cldnn::input_info("eltwise"), -1),
                cldnn::gemm("qkv_gemm", {cldnn::input_info("softmax"), cldnn::input_info("value_transposed")}, cldnn::data_types::f16, false, false),
                cldnn::permute("qkv_gemm_transposed", cldnn::input_info("qkv_gemm"), {0, 2, 1, 3}),
                cldnn::reorder("output_data", cldnn::input_info("qkv_gemm_transposed"), cldnn::format::bfyx, cldnn::data_types::f16),
                cldnn::reorder("scores_data", cldnn::input_info("softmax"), cldnn::format::bfyx, cldnn::data_types::f16));
        } else {
            topology.add(
                cldnn::input_layout("query", query_layout),
                cldnn::input_layout("key", key_layout),
                cldnn::input_layout("value", value_layout),
                cldnn::data("mask", mask_mem),
                cldnn::data("scale", scale_mem),
                cldnn::permute("query_transposed", cldnn::input_info("query"), {1, 2, 0, 3}),
                cldnn::permute("key_transposed", cldnn::input_info("key"), {1, 2, 3, 0}),
                cldnn::permute("value_transposed", cldnn::input_info("value"), {1, 2, 0, 3}),
                cldnn::gemm("qk_gemm", {cldnn::input_info("query_transposed"), cldnn::input_info("key_transposed")}, cldnn::data_types::f16, false, false),
                cldnn::eltwise("scale_div", {cldnn::input_info("qk_gemm"), cldnn::input_info("scale")}, cldnn::eltwise_mode::prod),
                cldnn::eltwise("eltwise", {cldnn::input_info("scale_div"), cldnn::input_info("mask")}, cldnn::eltwise_mode::sum),
                cldnn::softmax("softmax", cldnn::input_info("eltwise"), -1),
                cldnn::gemm("qkv_gemm", {cldnn::input_info("softmax"), cldnn::input_info("value_transposed")}, cldnn::data_types::f16, false, false),
                cldnn::reshape("qkv_gemm_reshape", cldnn::input_info("qkv_gemm"), {1, num_heads, v_head_size, num_queries}),
                cldnn::permute("qkv_gemm_transposed", cldnn::input_info("qkv_gemm_reshape"), {0, 2, 1, 3}),
                cldnn::reorder("output_data", cldnn::input_info("qkv_gemm_transposed"), cldnn::format::bfyx, cldnn::data_types::f16),
                cldnn::reorder("scores_data", cldnn::input_info("softmax"), cldnn::format::bfyx, cldnn::data_types::f16));
        }

        ov::intel_gpu::ExecutionConfig config = tests::get_test_default_config(test_engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        cldnn::network::ptr network = tests::get_network(test_engine, topology, config, tests::get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);

        auto outputs = network->execute();

        auto output_data_mem = outputs.at("output_data").get_memory();
        auto output_scores_mem = outputs.at("scores_data").get_memory();

        return {get_output_data_vec(output_data_mem, num_queries, v_head_size, num_heads),
                get_output_scores_vec(output_scores_mem, window_size, num_queries, num_keys, num_heads)};
    }

    std::vector<ov::float16> get_output_scores_vec(cldnn::memory::ptr scores_output, int window_size, int num_queries, int num_keys, int num_heads) {
        OPENVINO_ASSERT(scores_output->count() == static_cast<size_t>(num_heads * num_queries * num_keys));

        std::vector<ov::float16> output_scores(num_keys, 0);
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(scores_output, test_stream);
        for (int row_idx = 0; row_idx < window_size; row_idx++) {
            for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                for (int score_idx = 0; score_idx < num_keys; score_idx++) {
                    auto scores_offset = head_idx * num_queries * num_keys + (num_queries - window_size + row_idx) * num_keys + score_idx;
                    output_scores[score_idx] += mem_ptr[scores_offset];
                }
            }
        }

        return output_scores;
    }

    std::vector<ov::float16> get_output_data_vec(cldnn::memory::ptr data_output, int num_queries, int k_head_size, int num_heads) {
        OPENVINO_ASSERT(data_output->count() == static_cast<size_t>(num_queries * num_heads * k_head_size));

        std::vector<ov::float16> output_data(data_output->count());
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(data_output, test_stream);
        for (size_t i = 0; i < data_output->count(); i++)
            output_data[i] = mem_ptr[i];

        return output_data;
    }

    cldnn::memory::ptr get_mask_mem_combined_multi_head(int num_queries,
                                                        int num_keys,
                                                        int num_heads,
                                                        int num_kv_heads,
                                                        int sliding_window_size,
                                                        const ov::reference::XAttentionRetainedBlockIndicesForAllHeads& retained_blocks,
                                                        int block_size,
                                                        const std::vector<uint8_t>* qq_bias) {
        int heads_per_kv = num_heads / num_kv_heads;

        ov::PartialShape mask_shape;
        if (retained_blocks.empty()) {
            mask_shape = ov::PartialShape{1, 1, num_queries, num_keys};
        } else if (num_heads == num_kv_heads) {
            mask_shape = ov::PartialShape{1, num_heads, num_queries, num_keys};
        } else {
            mask_shape = ov::PartialShape{num_kv_heads, heads_per_kv, num_queries, num_keys};
        }

        auto mask_layout = cldnn::layout{mask_shape, cldnn::data_types::f16, cldnn::format::bfyx};
        auto mask_mem = test_engine.allocate_memory(mask_layout);
        cldnn::mem_lock<ov::float16> mem_ptr(mask_mem, test_stream);

        size_t total_elems = mask_layout.count();
        for (size_t i = 0; i < total_elems; ++i)
            mem_ptr[i] = std::numeric_limits<ov::float16>::lowest();
        if (retained_blocks.empty()) {
            if (sliding_window_size == 0) {
                int past_len = num_keys - num_queries + 1;
                for (int i = 0; i < num_queries; i++) {
                    for (int j = 0; j < num_keys; j++) {
                        mem_ptr[i * num_keys + j] = j >= past_len + i ? std::numeric_limits<ov::float16>::lowest() : ov::float16(0.f);
                    }
                }
            } else {
                int sliding_left = num_keys - num_queries - sliding_window_size + 1;
                int past_len = num_keys - num_queries + 1;

                for (int i = 0; i < num_queries; i++) {
                    for (int j = 0; j < num_keys; j++) {
                        bool is_min;
                        if (num_queries == num_keys) {
                            is_min = (j >= sliding_left + i) && (j <= i) ? 0 : 1;
                        } else {
                            is_min = (j >= sliding_left + i) && (j < past_len + i) ? 0 : 1;
                        }

                        mem_ptr[i * num_keys + j] = is_min ? std::numeric_limits<ov::float16>::lowest() : ov::float16(0.f);
                    }
                }
            }
        } else {
            for (int h = 0; h < num_heads; ++h) {
                int kv_idx = (num_heads == num_kv_heads) ? 0 : (h / heads_per_kv);
                int head_in_kv = (num_heads == num_kv_heads) ? h : (h % heads_per_kv);

                size_t head_offset = (static_cast<size_t>(kv_idx) * heads_per_kv + static_cast<size_t>(head_in_kv)) * static_cast<size_t>(num_queries) *
                                     static_cast<size_t>(num_keys);

                for (int i = 0; i < num_queries; i++) {
                    int left_idx = 0;
                    int right_idx = 0;

                    if (sliding_window_size == 0) {
                        int past_len = num_keys - num_queries + 1;
                        right_idx = past_len + i - 1;
                        left_idx = 0;
                    } else {
                        int sliding_left = num_keys - num_queries - sliding_window_size + 1;
                        int past_len = num_keys - num_queries + 1;
                        if (num_queries == num_keys) {
                            left_idx = sliding_left + i;
                            right_idx = i;
                        } else {
                            left_idx = sliding_left + i;
                            right_idx = past_len + i - 1;
                        }
                    }

                    left_idx = std::max(0, left_idx);
                    right_idx = std::min(num_keys - 1, right_idx);

                    for (const auto& [q_block_idx, k_block_idx] : retained_blocks[h]) {
                        int q_start = q_block_idx * block_size;
                        int q_end = std::min(q_start + block_size, num_queries);
                        int k_start = k_block_idx * block_size;
                        int k_end = std::min(k_start + block_size, num_keys);

                        if (i < q_start || i >= q_end)
                            continue;

                        for (int j = k_start; j < k_end; j++) {
                            if (j >= left_idx && j <= right_idx) {
                                mem_ptr[head_offset + i * num_keys + j] = ov::float16(0.f);
                            }
                        }
                    }
                }
            }
        }

        if (qq_bias && !qq_bias->empty()) {
            OPENVINO_ASSERT(qq_bias->size() == static_cast<size_t>(num_queries * num_queries));

            auto apply_mask_for_head = [&](size_t head_offset) {
                for (int i = 0; i < num_queries; i++) {
                    for (int j = 0; j < num_queries; j++) {
                        if (!(*qq_bias)[static_cast<size_t>(i) * static_cast<size_t>(num_queries) + static_cast<size_t>(j)]) {
                            mem_ptr[head_offset + static_cast<size_t>(i) * static_cast<size_t>(num_keys) + (num_keys - num_queries) + static_cast<size_t>(j)] =
                                std::numeric_limits<ov::float16>::lowest();
                        }
                    }
                }
            };

            if (retained_blocks.empty()) {
                apply_mask_for_head(0);
            } else {
                for (int h = 0; h < num_heads; ++h) {
                    int kv_idx = (num_heads == num_kv_heads) ? 0 : (h / heads_per_kv);
                    int head_in_kv = (num_heads == num_kv_heads) ? h : (h % heads_per_kv);
                    size_t head_offset = (static_cast<size_t>(kv_idx) * static_cast<size_t>(heads_per_kv) + static_cast<size_t>(head_in_kv)) *
                                         static_cast<size_t>(num_queries) * static_cast<size_t>(num_keys);
                    apply_mask_for_head(head_offset);
                }
            }
        }

        return mask_mem;
    }

    void rotate_block(std::vector<ov::float16>& cache_data,
                      std::vector<int> rotation_deltas,
                      std::vector<ov::float16> rotation_trig_lut_mem,
                      int rotated_block_idx,
                      int subsequence_rotated_block_idx,
                      int num_heads,
                      int k_head_size,
                      int block_size,
                      bool per_block) {
        // cache_data shape: [1, num_tokens, num_heads, k_head_size]
        int start_token_idx = subsequence_rotated_block_idx * block_size;

        for (int token_idx = 0; token_idx < block_size; token_idx++) {
            auto rotation_deltas_offset = per_block ? rotated_block_idx : rotated_block_idx * block_size + token_idx;
            auto rotation_trig_lut_idx = rotation_deltas[rotation_deltas_offset];
            for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                for (int k_head_size_idx = 0; k_head_size_idx < k_head_size / 2; k_head_size_idx++) {
                    auto input_offset = (start_token_idx + token_idx) * num_heads * k_head_size + head_idx * k_head_size + k_head_size_idx;

                    auto cache_value_0 = cache_data[input_offset];
                    auto cache_value_1 = cache_data[input_offset + k_head_size / 2];

                    ov::float16 rotation_value_cos = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx];
                    ov::float16 rotation_value_sin = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx + k_head_size / 2];

                    cache_data[input_offset] = cache_value_0 * rotation_value_cos - cache_value_1 * rotation_value_sin;
                    cache_data[input_offset + k_head_size / 2] = cache_value_0 * rotation_value_sin + cache_value_1 * rotation_value_cos;
                }
            }
        }
    }

public:
    std::vector<ov::float16> read_key_from_cache(cldnn::memory::ptr key_cache_mem, size_t seq_idx, int total_tokens) {
        // Read key vectors from key_cache memory
        // key_cache layout: [num_blocks, num_kv_heads, head_size, block_size]
        std::vector<ov::float16> key_data(pam.num_kv_heads * total_tokens * pam.k_head_size);

        const int blocks_start = pam.block_indices_begins[seq_idx];
        const int blocks_end = pam.block_indices_begins[seq_idx + 1];
        const int num_blocks = blocks_end - blocks_start;

        const bool is_compressed = pam.kv_cache_compression;

        if (!is_compressed) {
            // Uncompressed case: read as float16
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);

            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                const int physical_block = pam.block_indices[blocks_start + block_idx];
                const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                    const int token_idx = block_idx * pam.block_size + token_offset;

                    for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                        const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * pam.k_head_size * pam.block_size +
                                                  static_cast<size_t>(head_idx) * pam.k_head_size * pam.block_size;

                        const size_t output_base =
                            static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size + static_cast<size_t>(token_idx) * pam.k_head_size;

                        for (int dim = 0; dim < pam.k_head_size; dim++) {
                            const size_t cache_offset = cache_base + static_cast<size_t>(dim) * pam.block_size + token_offset;
                            key_data[output_base + dim] = cache_ptr[cache_offset];
                        }
                    }
                }
            }
        } else {
            if (pam.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                if (pam.is_int4_kv_cache()) {
                    // INT4 BY_CHANNEL: [num_blocks, kv_heads, k_head_size, block_size/2+4] u8
                    // block_size dim is packed: 2 u4 tokens per byte.
                    // Comp at [d, packed_block..packed_block+3]: 2 fp16 = inv_scale, zp per head dim.
                    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);
                    const int packed_block = pam.block_size / 2;
                    const int adj_block_size = packed_block + 4;  // block_size/2 + sizeof(fp16)*2

                    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                        const int physical_block = pam.block_indices[blocks_start + block_idx];
                        const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                        for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                            const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * pam.k_head_size * adj_block_size +
                                                      static_cast<size_t>(head_idx) * pam.k_head_size * adj_block_size;

                            for (int d = 0; d < pam.k_head_size; d++) {
                                // Read inv_scale and zp from comp region
                                const size_t comp_byte_off = cache_base + static_cast<size_t>(d) * adj_block_size + packed_block;
                                const ov::float16* comp = reinterpret_cast<const ov::float16*>(&cache_ptr[comp_byte_off]);
                                float inv_scale = static_cast<float>(comp[0]);
                                float zp_val = static_cast<float>(comp[1]);

                                for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                                    const int token_idx = block_idx * pam.block_size + token_offset;
                                    const size_t byte_off = cache_base + static_cast<size_t>(d) * adj_block_size + token_offset / 2;
                                    uint8_t packed_byte = cache_ptr[byte_off];
                                    uint8_t q = (token_offset % 2 == 0) ? (packed_byte & 0xFu) : ((packed_byte >> 4) & 0xFu);
                                    float dq = (static_cast<float>(q) - zp_val) * inv_scale;
                                    const size_t out_base =
                                        static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size + static_cast<size_t>(token_idx) * pam.k_head_size;
                                    key_data[out_base + d] = ov::float16(dq);
                                }
                            }
                        }
                    }
                } else {
                    // I8/U8 BY_CHANNEL: [num_blocks, num_kv_heads, head_size, block_size+4]
                    // Each dimension quantized across all tokens in block
                    cldnn::mem_lock<int8_t, cldnn::mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);
                    const int adj_block_size = pam.block_size + 4;

                    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                        const int physical_block = pam.block_indices[blocks_start + block_idx];
                        const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                        for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                            const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * pam.k_head_size * adj_block_size +
                                                      static_cast<size_t>(head_idx) * pam.k_head_size * adj_block_size;

                            for (int dim = 0; dim < pam.k_head_size; dim++) {
                                // Read scale and zero-point for this dimension
                                const size_t scale_offset = cache_base + static_cast<size_t>(dim) * adj_block_size + pam.block_size;
                                ov::float16 scale = *reinterpret_cast<const ov::float16*>(&cache_ptr[scale_offset]);
                                ov::float16 zp = *reinterpret_cast<const ov::float16*>(&cache_ptr[scale_offset + 2]);

                                // Dequantize all tokens for this dimension
                                for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                                    const int token_idx = block_idx * pam.block_size + token_offset;
                                    const size_t cache_offset = cache_base + static_cast<size_t>(dim) * adj_block_size + token_offset;
                                    const size_t output_offset =
                                        static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size + static_cast<size_t>(token_idx) * pam.k_head_size + dim;

                                    int8_t quantized_value = cache_ptr[cache_offset];
                                    float dequantized = (static_cast<float>(quantized_value) - static_cast<float>(zp)) * static_cast<float>(scale);
                                    key_data[output_offset] = ov::float16(dequantized);
                                }
                            }
                        }
                    }
                }
            } else {
                // BY_TOKEN
                if (pam.is_int4_kv_cache()) {
                    // INT4 BY_TOKEN: [num_blocks, kv_heads, k_head_size/2+8, block_size] u8
                    // Packing (SUBGROUP_SIZE=16):
                    //   Y = pack_group*16 + sglid, where pack_group = d/(2*16), sglid = d%16
                    //   lower nibble = q(dim[pack_group*32 + sglid])
                    //   upper nibble = q(dim[pack_group*32 + sglid + 16])
                    // Scale/ZP: fp16 in comp region at base + packed_head_size*block_size
                    //   inv_scale[t] at comp_ptr[t], zp[t] at comp_ptr[block_size + t]
                    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);
                    const int packed_head_size = pam.k_head_size / 2;
                    const int adj_head_size = packed_head_size + 8;
                    constexpr int SG = 16;

                    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                        const int physical_block = pam.block_indices[blocks_start + block_idx];
                        const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                        for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                            const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * adj_head_size * pam.block_size +
                                                      static_cast<size_t>(head_idx) * adj_head_size * pam.block_size;
                            // Scale and ZP are in the comp region after the packed data
                            const size_t comp_byte_base = cache_base + static_cast<size_t>(packed_head_size) * pam.block_size;
                            const auto* inv_scale_arr = reinterpret_cast<const ov::float16*>(&cache_ptr[comp_byte_base]);
                            const auto* zp_arr = reinterpret_cast<const ov::float16*>(&cache_ptr[comp_byte_base + pam.block_size * 2]);

                            for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                                const int token_idx = block_idx * pam.block_size + token_offset;
                                float inv_scale = static_cast<float>(inv_scale_arr[token_offset]);
                                float zp_val = static_cast<float>(zp_arr[token_offset]);
                                const size_t out_base =
                                    static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size + static_cast<size_t>(token_idx) * pam.k_head_size;

                                for (int d = 0; d < pam.k_head_size; d++) {
                                    int sglid_val = d % SG;
                                    int pack_group = d / (2 * SG);
                                    int group_in_pack = (d / SG) % 2;  // 0=lower nibble, 1=upper nibble
                                    int y = pack_group * SG + sglid_val;
                                    const size_t byte_off = cache_base + static_cast<size_t>(y) * pam.block_size + token_offset;
                                    uint8_t packed_byte = cache_ptr[byte_off];
                                    uint8_t q_d = (group_in_pack == 0) ? (packed_byte & 0xFu) : ((packed_byte >> 4) & 0xFu);
                                    key_data[out_base + d] = ov::float16((static_cast<float>(q_d) - zp_val) * inv_scale);
                                }
                            }
                        }
                    }

                    return key_data;
                }

                // BY_TOKEN: [num_blocks, num_kv_heads, head_size+4, block_size]
                // Token-wise quantization with shared scale/zp per token
                // Layout: data rows [0..head_size-1], scale at [head_size], zp at [head_size+2] (fp16)
                cldnn::mem_lock<int8_t, cldnn::mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    const int physical_block = pam.block_indices[blocks_start + block_idx];
                    const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                    for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                        const int token_idx = block_idx * pam.block_size + token_offset;

                        for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                            const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * (pam.k_head_size + 4) * pam.block_size +
                                                      static_cast<size_t>(head_idx) * (pam.k_head_size + 4) * pam.block_size;

                            // Read scale and zero-point for this token
                            // Scale is at [head_size][token], ZP is at [head_size+2][token] (2 rows below)
                            // token_offset * 2 because each half is 2 bytes (token 0 at offset 0-1, token 1 at offset 2-3, etc.)
                            const size_t scale_offset = cache_base + static_cast<size_t>(pam.k_head_size) * pam.block_size + token_offset * 2;
                            const size_t zp_offset = scale_offset + 2 * pam.block_size;  // ZP is 2 rows below scale
                            ov::float16 scale = *reinterpret_cast<const ov::float16*>(&cache_ptr[scale_offset]);
                            ov::float16 zp = *reinterpret_cast<const ov::float16*>(&cache_ptr[zp_offset]);

                            const size_t output_base =
                                static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size + static_cast<size_t>(token_idx) * pam.k_head_size;

                            // Dequantize all dimensions for this token
                            for (int dim = 0; dim < pam.k_head_size; dim++) {
                                const size_t cache_offset = cache_base + static_cast<size_t>(dim) * pam.block_size + token_offset;

                                int8_t quantized_value = cache_ptr[cache_offset];
                                float dequantized = (static_cast<float>(quantized_value) - static_cast<float>(zp)) * static_cast<float>(scale);

                                key_data[output_base + dim] = ov::float16(dequantized);
                            }
                        }
                    }
                }
            }  // end else (i8 BY_TOKEN)
        }  // is_compressed

        return key_data;
    }

private:
    std::vector<ov::float16> compute_diversity_reference(cldnn::memory::ptr key_cache_mem) {
        std::vector<ov::float16> diversity_output;

        for (size_t seq_idx = 0; seq_idx < pam.subsequence_descs.size(); seq_idx++) {
            const auto start_size = pam.adaptive_rkv_start_size;
            const auto evictable_size = pam.adaptive_rkv_evictable_sizes[seq_idx];

            // Read key data from key_cache instead of original key_data
            const auto& subsequence_desc = pam.subsequence_descs[seq_idx];
            const auto total_tokens = subsequence_desc.num_tokens + subsequence_desc.past_len;

            // Extract key vectors from key_cache memory
            std::vector<ov::float16> key_data = read_key_from_cache(key_cache_mem, seq_idx, total_tokens);

            ov::Shape key_shape = {static_cast<size_t>(pam.num_kv_heads), static_cast<size_t>(total_tokens), static_cast<size_t>(pam.k_head_size)};

            // Use reference implementation
            ov::reference::AdaptiveRKVDiversityCalculator<ov::float16> calculator(start_size, evictable_size, pam.block_size);

            auto block_diversity = calculator.calculate_block_diversity(key_data.data(), key_shape);

            const size_t num_evictable_blocks = static_cast<size_t>(evictable_size) / static_cast<size_t>(pam.block_size);
            // Flatten 2D to 1D: [num_evictable_blocks, evictable_size] -> [num_evictable_blocks * evictable_size]
            for (size_t block_idx = 0; block_idx < num_evictable_blocks; block_idx++) {
                for (size_t token_idx = 0; token_idx < static_cast<size_t>(evictable_size); token_idx++) {
                    diversity_output.push_back(block_diversity[block_idx][token_idx]);
                }
            }
        }

        return diversity_output;
    }

    PagedAttentionManager& pam;
    cldnn::engine& test_engine;
    cldnn::stream& test_stream;
};

template <typename T>
struct PagedAttentionTest : public ::testing::TestWithParam<T> {
public:
    tests::random_generator rg;
    cldnn::engine& engine = tests::get_test_engine();
    float tolerance = 2e-3;
    cldnn::memory::ptr last_key_cache_mem = nullptr;
    cldnn::memory::ptr last_output_data_mem = nullptr;
    std::vector<int> last_block_indices;
    std::vector<int> last_block_indices_begins;
    std::optional<PagedAttentionManager> pam;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);

        auto p = this->GetParam();

        pam.emplace(rg,
                    tests::get_test_engine(),
                    tests::get_test_stream(),
                    p.subsequences,
                    p.num_heads,
                    p.num_kv_heads,
                    p.k_head_size,
                    p.v_head_size,
                    p.block_size,
                    p.sliding_window_size,
                    p.kv_cache_compression,
                    p.key_cache_quant_mode,
                    p.scores_mode == ScoresMode::SNAPKV,
                    p.has_xattention,
                    p.rotation_config,
                    p.kv_cache_precision);
    }

    std::vector<ov::float16> get_output_data() {
        OPENVINO_ASSERT(last_output_data_mem != nullptr, "No output data available");
        std::vector<ov::float16> result(last_output_data_mem->count());
        cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(last_output_data_mem, tests::get_test_stream());
        for (size_t i = 0; i < last_output_data_mem->count(); i++)
            result[i] = mem_ptr[i];
        return result;
    }

    static std::vector<ov::float16> to_float16(const std::vector<float>& data) {
        std::vector<ov::float16> result(data.size());
        std::transform(data.begin(), data.end(), result.begin(), [](float value) {
            return ov::float16(value);
        });
        return result;
    }

    struct gpu_outputs {
        std::map<cldnn::primitive_id, cldnn::network_output> outputs;
        cldnn::memory::ptr key_cache_mem;
    };

    gpu_outputs run_gpu_inference(PagedAttentionManager& pam, T& p) {
        gpu_outputs result;

        ov::element::Type kv_cache_precision = p.kv_cache_precision;

        if (p.has_xattention) {
            pam.xattention_block_size.clear();
            if (p.xattention_block_size.has_value()) {
                pam.xattention_block_size = p.xattention_block_size.value();
            }
            pam.xattention_threshold.clear();
            if (p.xattention_threshold.has_value()) {
                pam.xattention_threshold.reserve(p.xattention_threshold->size());
                for (const float t : p.xattention_threshold.value()) {
                    pam.xattention_threshold.emplace_back(static_cast<ov::float16>(t));
                }
            }
            // Keep xattention_stride non-empty and per-sequence, reducing the mismatch risk with always-bound stride input.
            pam.xattention_stride.assign(p.subsequences.size(), 16);
        }

        if (p.has_adaptive_rkv) {
            pam.adaptive_rkv_diversity_block_set_indices_begins.push_back(0);
            pam.adaptive_rkv_start_size = p.start_size;

            for (size_t i = 0; i < p.subsequences.size(); i++) {
                // Use per-sequence evictable_size if available, otherwise use first value
                int evictable_size = i < p.evictable_sizes.size() ? p.evictable_sizes[i] : p.evictable_sizes[0];

                pam.adaptive_rkv_evictable_sizes.push_back(evictable_size);

                int start_block = p.start_size / p.block_size;
                int evictable_blocks = evictable_size / p.block_size;
                int global_start_block_idx = pam.block_indices_begins[i] + start_block;

                for (int b = 0; b < evictable_blocks; b++) {
                    pam.adaptive_rkv_diversity_block_set_indices.push_back(global_start_block_idx + b);
                }

                int prev_begin = pam.adaptive_rkv_diversity_block_set_indices_begins.back();
                pam.adaptive_rkv_diversity_block_set_indices_begins.push_back(prev_begin + evictable_blocks);
            }
        }

        if (p.has_qq_bias) {
            pam.qq_bias = p.qq_bias_config.qq_bias;
            pam.qq_bias_begins = p.qq_bias_config.qq_bias_begins;
        }

        if (p.token_type_ids.has_value()) {
            pam.token_type_ids = p.token_type_ids.value();
            EXPECT_EQ(pam.token_type_ids.size(), static_cast<size_t>(pam.subsequence_descs.back().num_tokens + pam.subsequence_descs.back().past_len));
        }

        if (p.has_sink_input && p.sink_values.has_value()) {
            pam.sinks = p.sink_values.value();
        }

        if (p.kv_cache_compression) {
            // INT4 quantization has larger error than INT8 (~17x larger step size)
            tolerance = (kv_cache_precision == ov::element::u4 || kv_cache_precision == ov::element::i4) ? 75e-3 : 25e-3;
        }

        auto query_mem = pam.get_query_memory();
        auto key_mem = pam.get_key_memory();
        auto value_mem = pam.get_value_memory();

        if (p.has_xattention) {
            result.key_cache_mem = pam.get_key_cache_memory_cm();
        } else {
            result.key_cache_mem = pam.get_key_cache_memory();
        }
        auto value_cache_mem = pam.get_value_cache_memory();

        auto past_lens_mem = pam.get_past_lens_memory();
        auto subsequence_begins_mem = pam.get_subsequence_begins_memory();
        auto block_indices_mem = pam.get_block_indices_memory();
        auto block_indices_begins_mem = pam.get_block_indices_begins_memory();

        auto scale_mem = pam.get_scale_memory();
        auto sliding_window_mem = pam.get_sliding_window_memory();
        auto alibi_mem = pam.get_alibi_memory();
        auto max_context_len_mem = pam.get_max_context_len_memory();

        // scores calculation related memory buffers
        auto score_aggregation_mem = pam.get_score_aggregation();

        // cache rotation related memory buffers
        auto rotated_block_indices_mem = pam.get_rotated_block_indices_memory();
        auto rotation_deltas_mem = pam.get_rotation_deltas_memory();
        auto rotation_trig_lut_mem = pam.get_rotation_trig_lut_memory();

        auto xattention_threshold_mem = pam.get_xattention_threshold_memory();
        auto xattention_block_size_mem = pam.get_xattention_block_size_memory();
        auto xattention_stride_mem = pam.get_xattention_stride_memory();
        auto sinks_mem = pam.get_sinks_memory();
        auto adaptive_rkv_start_size_mem = pam.get_adaptive_rkv_start_size_memory();
        auto adaptive_rkv_evictable_sizes_mem = pam.get_adaptive_rkv_evictable_sizes_memory();
        auto adaptive_rkv_diversity_block_set_indices_mem = pam.get_adaptive_rkv_diversity_block_set_indices_memory();
        auto adaptive_rkv_diversity_block_set_indices_begins_mem = pam.get_adaptive_rkv_diversity_block_set_indices_begins_memory();
        auto token_type_ids_mem = pam.get_token_type_ids_memory();

        auto qq_bias = pam.get_qq_bias_memory();
        auto qq_bias_begins = pam.get_qq_bias_begins_memory();
        auto query_layout = query_mem->get_layout();
        auto key_layout = key_mem->get_layout();
        auto value_layout = value_mem->get_layout();
        auto key_cache_layout = result.key_cache_mem->get_layout();
        auto value_cache_layout = value_cache_mem->get_layout();
        auto past_lens_layout = past_lens_mem->get_layout();
        auto subsequence_begins_layout = subsequence_begins_mem->get_layout();
        auto block_indices_layout = block_indices_mem->get_layout();
        auto block_indices_begins_layout = block_indices_begins_mem->get_layout();
        auto scale_layout = scale_mem->get_layout();
        auto sliding_window_layout = sliding_window_mem->get_layout();
        auto alibi_layout = alibi_mem->get_layout();
        auto max_context_len_layout = max_context_len_mem->get_layout();
        auto score_aggregation_window_layout = score_aggregation_mem->get_layout();
        auto rotated_block_indices_layout = rotated_block_indices_mem->get_layout();
        auto rotation_deltas_layout = rotation_deltas_mem->get_layout();
        auto rotation_trig_lut_layout = rotation_trig_lut_mem->get_layout();
        auto xattention_threshold_layout = xattention_threshold_mem->get_layout();
        auto xattention_block_size_layout = xattention_block_size_mem->get_layout();
        auto xattention_stride_layout = xattention_stride_mem->get_layout();
        auto sinks_layout = sinks_mem->get_layout();
        auto adaptive_rkv_start_size_layout = adaptive_rkv_start_size_mem->get_layout();
        auto adaptive_rkv_evictable_sizes_layout = adaptive_rkv_evictable_sizes_mem->get_layout();
        auto adaptive_rkv_diversity_block_set_indices_layout = adaptive_rkv_diversity_block_set_indices_mem->get_layout();
        auto adaptive_rkv_diversity_block_set_indices_begins_layout = adaptive_rkv_diversity_block_set_indices_begins_mem->get_layout();
        auto token_type_ids_layout = token_type_ids_mem->get_layout();
        auto qq_bias_layout = qq_bias->get_layout();
        auto qq_bias_begins_layout = qq_bias_begins->get_layout();

        // make layouts dynamic
        query_layout.set_partial_shape(ov::PartialShape{-1, p.num_heads * p.k_head_size});
        key_layout.set_partial_shape(ov::PartialShape{-1, p.num_kv_heads * p.k_head_size});
        value_layout.set_partial_shape(ov::PartialShape{-1, p.num_kv_heads * p.v_head_size});
        // key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.k_head_size, p.block_size });
        {
            auto pshape = key_cache_layout.get_partial_shape();
            pshape[0] = -1;
            key_cache_layout.set_partial_shape(pshape);
        }
        // value_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.block_size, p.v_head_size });
        {
            auto pshape = value_cache_layout.get_partial_shape();
            pshape[0] = -1;
            value_cache_layout.set_partial_shape(pshape);
        }
        past_lens_layout.set_partial_shape(ov::PartialShape{-1});
        subsequence_begins_layout.set_partial_shape(ov::PartialShape{-1});
        block_indices_layout.set_partial_shape(ov::PartialShape{-1});
        block_indices_begins_layout.set_partial_shape(ov::PartialShape{-1});
        score_aggregation_window_layout.set_partial_shape(ov::PartialShape{-1});
        rotated_block_indices_layout.set_partial_shape(ov::PartialShape{-1});
        rotation_deltas_layout.set_partial_shape(ov::PartialShape{-1, -1});
        rotation_trig_lut_layout.set_partial_shape(ov::PartialShape{-1, p.k_head_size});
        xattention_threshold_layout.set_partial_shape(ov::PartialShape{-1});
        adaptive_rkv_evictable_sizes_layout.set_partial_shape(ov::PartialShape{-1});
        adaptive_rkv_diversity_block_set_indices_layout.set_partial_shape(ov::PartialShape{-1});
        adaptive_rkv_diversity_block_set_indices_begins_layout.set_partial_shape(ov::PartialShape{-1});
        qq_bias_layout.set_partial_shape(ov::PartialShape{-1});
        qq_bias_begins_layout.set_partial_shape(ov::PartialShape{-1});

        if (p.dynamic_paddings) {
            const auto padding_axis = 1;
            const auto pad_before = p.k_head_size;
            const auto pad_after = p.k_head_size * 2;

            query_layout.data_padding._dynamic_dims_mask[padding_axis] = 1;

            auto query_data_layout = query_mem->get_layout();
            auto padded_query_data_layout = query_data_layout;
            padded_query_data_layout.data_padding._lower_size[padding_axis] = pad_before;
            padded_query_data_layout.data_padding._upper_size[padding_axis] = pad_after;

            auto new_query_memory = tests::get_test_engine().allocate_memory(padded_query_data_layout, false);

            cldnn::mem_lock<ov::float16> query_mem_lock(query_mem, tests::get_test_stream());
            cldnn::mem_lock<ov::float16> new_query_mem_lock(new_query_memory, tests::get_test_stream());

            auto query_data_shape = query_data_layout.get_shape();
            for (size_t b = 0; b < query_data_shape[0]; b++) {
                for (size_t f = 0; f < query_data_shape[1]; f++) {
                    auto input_offset = query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));
                    auto output_offset =
                        padded_query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));

                    new_query_mem_lock[output_offset] = query_mem_lock[input_offset];
                }
            }
            query_mem = new_query_memory;
        }

        std::vector<cldnn::input_info> pa_inputs = {cldnn::input_info("query"),
                                                    cldnn::input_info("key"),
                                                    cldnn::input_info("value"),
                                                    cldnn::input_info("key_cache"),
                                                    cldnn::input_info("value_cache"),
                                                    cldnn::input_info("past_lens"),
                                                    cldnn::input_info("subsequence_begins"),
                                                    cldnn::input_info("block_indices"),
                                                    cldnn::input_info("block_indices_begins"),
                                                    cldnn::input_info("scale"),
                                                    cldnn::input_info("sliding_window"),
                                                    cldnn::input_info("alibi"),
                                                    cldnn::input_info("max_context_len"),
                                                    cldnn::input_info("score_aggregation_window"),
                                                    cldnn::input_info("rotated_block_indices"),
                                                    cldnn::input_info("rotation_deltas"),
                                                    cldnn::input_info("rotation_trig_lut_modified"),
                                                    cldnn::input_info("xattention_threshold"),
                                                    cldnn::input_info("xattention_block_size"),
                                                    cldnn::input_info("xattention_stride"),
                                                    cldnn::input_info("sinks"),
                                                    cldnn::input_info("adaptive_rkv_start_size"),
                                                    cldnn::input_info("adaptive_rkv_evictable_sizes"),
                                                    cldnn::input_info("adaptive_rkv_diversity_block_set_indices"),
                                                    cldnn::input_info("adaptive_rkv_diversity_block_set_indices_begins"),
                                                    cldnn::input_info("token_type_ids"),
                                                    cldnn::input_info("qq_bias"),
                                                    cldnn::input_info("qq_bias_begins")};

        auto pa_prim = cldnn::paged_attention("paged_attention", pa_inputs);

        pa_prim.k_head_size = p.k_head_size;
        pa_prim.v_head_size = p.v_head_size;
        pa_prim.kv_heads_num = p.num_kv_heads;
        pa_prim.heads_num = p.num_heads;
        pa_prim.scale_val = pam.get_default_scale();
        pa_prim.has_alibi = false;
        pa_prim.has_token_type_ids = p.token_type_ids.has_value() || p.has_sink_input;
        pa_prim.has_sink_input = p.has_sink_input;

        int num_outputs = 1;
        if (p.scores_mode != ScoresMode::DISABLED)
            num_outputs++;
        if (p.has_adaptive_rkv)
            num_outputs++;
        pa_prim.num_outputs = num_outputs;
        pa_prim.has_rotated_blocks = p.rotation_config.apply_rotation;
        pa_prim.has_score_aggregation = p.scores_mode == ScoresMode::SNAPKV;
        pa_prim.has_adaptive_rkv = p.has_adaptive_rkv;
        pa_prim.sliding_window = p.sliding_window_size;
        pa_prim.is_key_by_channel = (p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);
        if (p.has_xattention) {
            pa_prim.has_xattention = true;
        }

        pa_prim.has_qq_bias = p.has_qq_bias;

        cldnn::topology topology;

        topology.add(cldnn::input_layout("query", query_layout),
                     cldnn::input_layout("key", key_layout),
                     cldnn::input_layout("value", value_layout),
                     cldnn::input_layout("key_cache", key_cache_layout),
                     cldnn::input_layout("value_cache", value_cache_layout),
                     cldnn::input_layout("past_lens", past_lens_layout),
                     cldnn::input_layout("subsequence_begins", subsequence_begins_layout),
                     cldnn::input_layout("block_indices", block_indices_layout),
                     cldnn::input_layout("block_indices_begins", block_indices_begins_layout),
                     cldnn::input_layout("scale", scale_layout),
                     cldnn::input_layout("sliding_window", sliding_window_layout),
                     cldnn::input_layout("alibi", alibi_layout),
                     cldnn::input_layout("max_context_len", max_context_len_layout),
                     cldnn::input_layout("score_aggregation_window", score_aggregation_window_layout),
                     pa_prim,
                     cldnn::reorder("output_data", cldnn::input_info("paged_attention", 0), cldnn::format::bfyx, cldnn::data_types::f16));

        int output_idx = 1;
        if (p.scores_mode != ScoresMode::DISABLED) {
            topology.add(cldnn::reorder("output_scores", cldnn::input_info("paged_attention", output_idx), cldnn::format::bfyx, cldnn::data_types::f16));
            output_idx++;
        }
        if (p.has_adaptive_rkv) {
            topology.add(cldnn::reorder("output_diversity", cldnn::input_info("paged_attention", output_idx), cldnn::format::bfyx, cldnn::data_types::f16));
        }

        {
            topology.add(cldnn::input_layout("rotated_block_indices", rotated_block_indices_layout));
            topology.add(cldnn::input_layout("rotation_deltas", rotation_deltas_layout));
            topology.add(cldnn::input_layout("rotation_trig_lut", rotation_trig_lut_layout));

            // add dummy activation operation to simulate an empty PA `rotation_trig_lut` buffer for shapes like [0, k_head_size]
            topology.add(cldnn::activation("rotation_trig_lut_modified", cldnn::input_info("rotation_trig_lut"), cldnn::activation_func::none));

            topology.add(cldnn::input_layout("xattention_threshold", xattention_threshold_layout));
            topology.add(cldnn::input_layout("xattention_block_size", xattention_block_size_layout));
            topology.add(cldnn::input_layout("xattention_stride", xattention_stride_layout));
            topology.add(cldnn::input_layout("sinks", sinks_layout));

            topology.add(cldnn::input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_layout));
            topology.add(cldnn::input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_layout));
            topology.add(cldnn::input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_layout));
            topology.add(cldnn::input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_layout));
            topology.add(cldnn::input_layout("token_type_ids", token_type_ids_layout));
            topology.add(cldnn::input_layout("qq_bias", qq_bias_layout));
            topology.add(cldnn::input_layout("qq_bias_begins", qq_bias_begins_layout));
        }

        ov::intel_gpu::ExecutionConfig config = tests::get_test_default_config(tests::get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        // FlashAttn v1 or v2?
        config.set_property(ov::intel_gpu::could_use_flashattn_v2(p.force_flashattn_v2 ? true : p.disable_flashattn_v2));
        config.set_property(ov::internal::key_cache_quant_mode(p.key_cache_quant_mode));
        if (kv_cache_precision != ov::element::dynamic) {
            config.set_property(ov::hint::kv_cache_precision(kv_cache_precision));
        }
        cldnn::network::ptr network = tests::get_network(tests::get_test_engine(), topology, config, tests::get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);
        network->set_input_data("key_cache", result.key_cache_mem);
        network->set_input_data("value_cache", value_cache_mem);
        network->set_input_data("past_lens", past_lens_mem);
        network->set_input_data("subsequence_begins", subsequence_begins_mem);
        network->set_input_data("block_indices", block_indices_mem);
        network->set_input_data("block_indices_begins", block_indices_begins_mem);
        network->set_input_data("scale", scale_mem);
        network->set_input_data("sliding_window", sliding_window_mem);
        network->set_input_data("alibi", alibi_mem);
        network->set_input_data("max_context_len", max_context_len_mem);
        network->set_input_data("score_aggregation_window", score_aggregation_mem);
        network->set_input_data("rotated_block_indices", rotated_block_indices_mem);
        network->set_input_data("rotation_deltas", rotation_deltas_mem);
        network->set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
        network->set_input_data("xattention_threshold", xattention_threshold_mem);
        network->set_input_data("xattention_block_size", xattention_block_size_mem);

        network->set_input_data("xattention_stride", xattention_stride_mem);
        network->set_input_data("sinks", sinks_mem);
        network->set_input_data("adaptive_rkv_start_size", adaptive_rkv_start_size_mem);
        network->set_input_data("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_mem);
        network->set_input_data("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_mem);
        network->set_input_data("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_mem);
        network->set_input_data("token_type_ids", token_type_ids_mem);
        network->set_input_data("qq_bias", qq_bias);
        network->set_input_data("qq_bias_begins", qq_bias_begins);

        last_key_cache_mem = result.key_cache_mem;
        last_block_indices = pam.block_indices;
        last_block_indices_begins = pam.block_indices_begins;

        result.outputs = network->execute();

        last_output_data_mem = result.outputs.at("output_data").get_memory();

        return result;
    }

    void execute(T& p, bool run_reference = true) {
        ASSERT_TRUE(this->pam.has_value());
        auto& pam = *this->pam;

        auto result = run_gpu_inference(pam, p);

        if (!run_reference) {
            return;
        }

        cldnn::memory::ptr output_data_mem = last_output_data_mem;
        cldnn::memory::ptr output_scores_mem = nullptr;
        cldnn::memory::ptr output_diversity_mem = nullptr;

        output_data_mem = result.outputs.at("output_data").get_memory();
        if (p.scores_mode != ScoresMode::DISABLED) {
            output_scores_mem = result.outputs.at("output_scores").get_memory();
        }
        if (p.has_adaptive_rkv) {
            output_diversity_mem = result.outputs.at("output_diversity").get_memory();
        }

        // Verify KV cache was correctly written (CM PA path only)
        // NOTE: This verification is specific to CM PA layout and should NOT run for OCL micro_sdpa
        // because they use different layouts (key cache is [N,K,H,B] in OCL vs [N,K,B,H] in CM)
        verify_cm_kv_cache_write(p);

        auto ref_data = PagedAttentionReference(pam).get_reference(result.key_cache_mem);
        if (p.has_xattention) {
            compare_xattention(output_data_mem, output_scores_mem, ref_data, p.num_heads, p.k_head_size);
        } else {
            compare(output_data_mem, output_scores_mem, output_diversity_mem, ref_data);
        }
    }

    void compare(cldnn::memory::ptr data_output_mem,
                 cldnn::memory::ptr scores_output_mem,
                 cldnn::memory::ptr diversity_output_mem,
                 std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> ref_data) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), std::get<0>(ref_data).size());
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(data_output_mem, tests::get_test_stream());
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<0>(ref_data)[i], tolerance) << " at index=" << i;
            }
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), std::get<1>(ref_data).size());
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(scores_output_mem, tests::get_test_stream());
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<1>(ref_data)[i], tolerance) << " at index=" << i;
            }
        }

        if (diversity_output_mem) {
            ASSERT_EQ(diversity_output_mem->count(), std::get<2>(ref_data).size());
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(diversity_output_mem, tests::get_test_stream());
            // Relaxed tolerance due to float32 (GPU) vs float16 (reference) accumulator difference
            float diversity_tolerance = tolerance * 10.0f;
            for (size_t i = 0; i < diversity_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<2>(ref_data)[i], diversity_tolerance) << " at index=" << i;
            }
        }
    }

    void compare_xattention(cldnn::memory::ptr data_output_mem,
                            cldnn::memory::ptr scores_output_mem,
                            std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> ref_data,
                            size_t num_heads,
                            size_t head_size) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), std::get<0>(ref_data).size());
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(data_output_mem, tests::get_test_stream());
            int mismatch_count = 0;

#if XATTENTION_DEBUG_VERBOSE
            XAttentionErrorStats stats;
            collect_xattention_error_stats(mem_ptr.data(), std::get<0>(ref_data), num_heads, head_size, tolerance, stats);
            mismatch_count = stats.mismatch_count;

            // Print detailed statistics on failure
            if (mismatch_count > int(data_output_mem->count() * 0.04)) {
                print_xattention_error_details(stats, mem_ptr.data(), std::get<0>(ref_data), num_heads, head_size, tolerance);
            }
#else
            // Simple counting when verbose debug is disabled
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                float actual = static_cast<float>(mem_ptr[i]);
                float expected = static_cast<float>(std::get<0>(ref_data)[i]);
                float error = std::fabs(actual - expected);
                if (error > tolerance) {
                    mismatch_count++;
                }
            }
#endif

            EXPECT_LE(mismatch_count, int(data_output_mem->count() * 0.04));
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), std::get<1>(ref_data).size());
            cldnn::mem_lock<ov::float16, cldnn::mem_lock_type::read> mem_ptr(scores_output_mem, tests::get_test_stream());
            int mismatch_count = 0;
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                if (std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(std::get<1>(ref_data)[i])) > tolerance) {
                    mismatch_count++;
                }
            }
            EXPECT_LE(mismatch_count, int(scores_output_mem->count() * 0.04));
        }
    }

private:
    // Helper: Verify CM PA KV cache was correctly written (CM path only)
    void verify_cm_kv_cache_write(const T& p) {
        if (last_key_cache_mem == nullptr || !p.has_xattention)
            return;

        // Count total tokens for single-token BY_CHANNEL skip logic
        int total_new_tokens = 0;
        for (const auto& s : p.subsequences)
            total_new_tokens += s.num_tokens;

        constexpr int kv_sub_block_size = 16;
        const bool is_by_channel = p.kv_cache_compression && p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL;
        const bool is_by_token = p.kv_cache_compression && p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_TOKEN;

        // CM cache layout: [num_blocks, num_kv_heads, adjusted_block_size, adjusted_head_size]
        const int adj_head_size = is_by_token ? (p.k_head_size + 4) : p.k_head_size;
        const int adj_block_size = is_by_channel ? (p.block_size + (p.block_size / kv_sub_block_size) * 4) : p.block_size;
        const int elem_size = p.kv_cache_compression ? 1 : 2;
        const size_t head_region_bytes = static_cast<size_t>(adj_block_size) * adj_head_size * elem_size;
        const size_t block_stride_bytes = static_cast<size_t>(p.num_kv_heads) * head_region_bytes;
        const int token_data_stride = p.k_head_size * elem_size;

        cldnn::mem_lock<int8_t, cldnn::mem_lock_type::read> cache_lock(last_key_cache_mem, tests::get_test_stream());

        int missing_count = 0, nan_count = 0, inf_count = 0, zero_scale_count = 0, out_of_range_zp_count = 0;
        std::vector<std::tuple<int, int, int, int, int>> nan_locations, inf_locations;
        std::vector<std::tuple<int, int, int, int, float, float>> zero_scale_locations, out_of_range_zp_locations;

        int total_tokens = 0;
        for (int i = 0; i < static_cast<int>(p.subsequences.size()); i++) {
            const int past_len = p.subsequences[i].past_len;
            const int num_tokens = p.subsequences[i].num_tokens;
            for (int t = 0; t < num_tokens; t++) {
                total_tokens++;
                const int absolute_pos = past_len + t;
                const int block_idx = absolute_pos / p.block_size;
                const int token_in_block = absolute_pos % p.block_size;
                const int physical_block = last_block_indices[last_block_indices_begins[i] + block_idx];
                const size_t block_base = static_cast<size_t>(physical_block) * block_stride_bytes;

                for (int head = 0; head < p.num_kv_heads; head++) {
                    const size_t head_base = block_base + static_cast<size_t>(head) * head_region_bytes;
                    const size_t token_offset = head_base + static_cast<size_t>(token_in_block) * token_data_stride;

                    // Check for missing token write (all-zero data)
                    bool skip_zero_check = is_by_channel && (total_new_tokens <= 1);
                    if (!skip_zero_check) {
                        bool all_zero = true;
                        for (int b = 0; b < token_data_stride; b++) {
                            if (cache_lock[token_offset + b] != 0) {
                                all_zero = false;
                                break;
                            }
                        }
                        if (all_zero && head == 0) {
                            missing_count++;
                            GPU_DEBUG_LOG << "KV cache update MISSING: seq=" << i << " token=" << t << " (absolute_pos=" << absolute_pos
                                          << ") at byte_offset=" << token_offset << std::endl;
                        }
                    }

                    // Check for NaN/INF in data or scale/zp
                    check_kv_cache_nan_inf(cache_lock.data(),
                                           p,
                                           is_by_token,
                                           is_by_channel,
                                           head_base,
                                           token_in_block,
                                           i,
                                           t,
                                           absolute_pos,
                                           head,
                                           nan_count,
                                           inf_count,
                                           zero_scale_count,
                                           out_of_range_zp_count,
                                           nan_locations,
                                           inf_locations,
                                           zero_scale_locations,
                                           out_of_range_zp_locations);
                }
            }
        }

        report_kv_cache_issues(nan_count,
                               inf_count,
                               zero_scale_count,
                               out_of_range_zp_count,
                               nan_locations,
                               inf_locations,
                               zero_scale_locations,
                               out_of_range_zp_locations,
                               total_tokens,
                               p,
                               is_by_token,
                               is_by_channel);
        EXPECT_EQ(missing_count, 0) << missing_count << " out of " << total_tokens << " tokens were not written to key cache by KV update kernel";
        EXPECT_EQ(nan_count, 0) << "KV cache contains NaN values";
        EXPECT_EQ(inf_count, 0) << "KV cache contains INF values";
        EXPECT_EQ(zero_scale_count, 0) << "KV cache contains zero/near-zero scale values (causes division by zero in dequant)";
        // ZP can be outside [0, 255] legitimately - see pa-quantization skill for details
    }

    // Helper: Check for NaN/INF in KV cache token
    void check_kv_cache_nan_inf(const int8_t* cache_data,
                                const T& p,
                                bool is_by_token,
                                bool is_by_channel,
                                size_t head_base,
                                int token_in_block,
                                int seq_idx,
                                int token_idx,
                                int absolute_pos,
                                int head,
                                int& nan_count,
                                int& inf_count,
                                int& zero_scale_count,
                                int& out_of_range_zp_count,
                                std::vector<std::tuple<int, int, int, int, int>>& nan_locations,
                                std::vector<std::tuple<int, int, int, int, int>>& inf_locations,
                                std::vector<std::tuple<int, int, int, int, float, float>>& zero_scale_locations,
                                std::vector<std::tuple<int, int, int, int, float, float>>& out_of_range_zp_locations) {
        constexpr int kv_sub_block_size = 16;
        const size_t token_offset = head_base + static_cast<size_t>(token_in_block) * p.k_head_size * (p.kv_cache_compression ? 1 : 2);

        if (!p.kv_cache_compression) {
            // FP16 cache: check data for NaN/INF
            const ov::float16* fp16_ptr = reinterpret_cast<const ov::float16*>(cache_data + token_offset);
            for (int dim = 0; dim < p.k_head_size; dim++) {
                float val = static_cast<float>(fp16_ptr[dim]);
                if (std::isnan(val)) {
                    nan_count++;
                    if (nan_locations.size() < 10) {
                        nan_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, dim));
                    }
                }
                if (std::isinf(val)) {
                    inf_count++;
                    if (inf_locations.size() < 10) {
                        inf_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, dim));
                    }
                }
            }
        } else if (is_by_token) {
            // BY_TOKEN: check scale/zp
            const size_t scale_offset = head_base + static_cast<size_t>(p.block_size) * p.k_head_size + static_cast<size_t>(token_in_block) * 2;
            const size_t zp_offset = scale_offset + static_cast<size_t>(p.block_size) * 2;
            const ov::float16* scale_ptr = reinterpret_cast<const ov::float16*>(cache_data + scale_offset);
            const ov::float16* zp_ptr = reinterpret_cast<const ov::float16*>(cache_data + zp_offset);
            float scale = static_cast<float>(*scale_ptr);
            float zp = static_cast<float>(*zp_ptr);

            check_scale_zp_validity(scale,
                                    zp,
                                    seq_idx,
                                    token_idx,
                                    absolute_pos,
                                    head,
                                    -1,
                                    nan_count,
                                    inf_count,
                                    zero_scale_count,
                                    out_of_range_zp_count,
                                    nan_locations,
                                    inf_locations,
                                    zero_scale_locations,
                                    out_of_range_zp_locations);
        } else if (is_by_channel) {
            // BY_CHANNEL: check scale/zp per channel (only first 4 channels to limit overhead)
            int sub_block_idx = token_in_block / kv_sub_block_size;
            int group_num = p.block_size / kv_sub_block_size;
            for (int ch = 0; ch < std::min(p.k_head_size, 4); ch++) {
                size_t scale_offset =
                    head_base + static_cast<size_t>(p.block_size) * p.k_head_size + (static_cast<size_t>(sub_block_idx) * p.k_head_size + ch) * 2;
                size_t zp_offset = scale_offset + static_cast<size_t>(group_num) * p.k_head_size * 2;
                const ov::float16* scale_ptr = reinterpret_cast<const ov::float16*>(cache_data + scale_offset);
                const ov::float16* zp_ptr = reinterpret_cast<const ov::float16*>(cache_data + zp_offset);
                float scale = static_cast<float>(*scale_ptr);
                float zp = static_cast<float>(*zp_ptr);

                check_scale_zp_validity(scale,
                                        zp,
                                        seq_idx,
                                        token_idx,
                                        absolute_pos,
                                        head,
                                        ch,
                                        nan_count,
                                        inf_count,
                                        zero_scale_count,
                                        out_of_range_zp_count,
                                        nan_locations,
                                        inf_locations,
                                        zero_scale_locations,
                                        out_of_range_zp_locations);
            }
        }
    }

    // Helper: Check scale/zp for NaN/INF/zero
    void check_scale_zp_validity(float scale,
                                 float zp,
                                 int seq_idx,
                                 int token_idx,
                                 int absolute_pos,
                                 int head,
                                 int dim,
                                 int& nan_count,
                                 int& inf_count,
                                 int& zero_scale_count,
                                 int& out_of_range_zp_count,
                                 std::vector<std::tuple<int, int, int, int, int>>& nan_locations,
                                 std::vector<std::tuple<int, int, int, int, int>>& inf_locations,
                                 std::vector<std::tuple<int, int, int, int, float, float>>& zero_scale_locations,
                                 std::vector<std::tuple<int, int, int, int, float, float>>& out_of_range_zp_locations) {
        if (std::isnan(scale) || std::isnan(zp)) {
            nan_count += (std::isnan(scale) ? 1 : 0) + (std::isnan(zp) ? 1 : 0);
            if (nan_locations.size() < 10) {
                nan_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, dim));
            }
        }
        if (std::isinf(scale) || std::isinf(zp)) {
            inf_count += (std::isinf(scale) ? 1 : 0) + (std::isinf(zp) ? 1 : 0);
            if (inf_locations.size() < 10) {
                inf_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, dim));
            }
        }
        if (std::fabs(scale) < 1e-6f) {
            zero_scale_count++;
            if (zero_scale_locations.size() < 10) {
                zero_scale_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, scale, zp));
            }
        }
        if (zp < -1.0f || zp > 256.0f) {
            out_of_range_zp_count++;
            if (out_of_range_zp_locations.size() < 10) {
                out_of_range_zp_locations.push_back(std::make_tuple(seq_idx, token_idx, absolute_pos, head, scale, zp));
            }
        }
    }

    // Helper: Report KV cache verification issues
    void report_kv_cache_issues(int nan_count,
                                int inf_count,
                                int zero_scale_count,
                                int out_of_range_zp_count,
                                const std::vector<std::tuple<int, int, int, int, int>>& nan_locations,
                                const std::vector<std::tuple<int, int, int, int, int>>& inf_locations,
                                const std::vector<std::tuple<int, int, int, int, float, float>>& zero_scale_locations,
                                const std::vector<std::tuple<int, int, int, int, float, float>>& out_of_range_zp_locations,
                                int total_tokens,
                                const T& p,
                                bool is_by_token,
                                bool is_by_channel) {
        if (nan_count > 0) {
            GPU_DEBUG_LOG << "\nKV cache contains " << nan_count << " NaN values:" << std::endl;
            for (const auto& [seq, tok, abs_pos, h, d] : nan_locations) {
                GPU_DEBUG_LOG << "  seq=" << seq << " token=" << tok << " (absolute_pos=" << abs_pos << ") head=" << h;
                if (d == -1) {
                    GPU_DEBUG_LOG << " [BY_TOKEN scale/zp]";
                } else if (d >= 0) {
                    GPU_DEBUG_LOG << " [BY_CHANNEL scale/zp channel=" << d << "]";
                } else {
                    GPU_DEBUG_LOG << " [FP16 dim=" << d << "]";
                }
                GPU_DEBUG_LOG << std::endl;
            }
            if (nan_locations.size() < static_cast<size_t>(nan_count)) {
                GPU_DEBUG_LOG << "  ... and " << (nan_count - nan_locations.size()) << " more" << std::endl;
            }
        }
        if (inf_count > 0) {
            GPU_DEBUG_LOG << "\nKV cache contains " << inf_count << " INF values:" << std::endl;
            for (const auto& [seq, tok, abs_pos, h, d] : inf_locations) {
                GPU_DEBUG_LOG << "  seq=" << seq << " token=" << tok << " (absolute_pos=" << abs_pos << ") head=" << h;
                if (d == -1) {
                    GPU_DEBUG_LOG << " [BY_TOKEN scale/zp]";
                } else if (d >= 0) {
                    GPU_DEBUG_LOG << " [BY_CHANNEL scale/zp channel=" << d << "]";
                } else {
                    GPU_DEBUG_LOG << " [FP16 dim=" << d << "]";
                }
                GPU_DEBUG_LOG << std::endl;
            }
            if (inf_locations.size() < static_cast<size_t>(inf_count)) {
                GPU_DEBUG_LOG << "  ... and " << (inf_count - inf_locations.size()) << " more" << std::endl;
            }
        }
        if (zero_scale_count > 0) {
            GPU_DEBUG_LOG << "\nKV cache contains " << zero_scale_count << " zero/near-zero scale values:" << std::endl;
            for (const auto& [seq, tok, abs_pos, h, scale, zp] : zero_scale_locations) {
                GPU_DEBUG_LOG << "  seq=" << seq << " token=" << tok << " (absolute_pos=" << abs_pos << ") head=" << h << " scale=" << scale << " zp=" << zp
                              << std::endl;
            }
            if (zero_scale_locations.size() < static_cast<size_t>(zero_scale_count)) {
                GPU_DEBUG_LOG << "  ... and " << (zero_scale_count - zero_scale_locations.size()) << " more" << std::endl;
            }
        }
        if (out_of_range_zp_count > 0) {
            GPU_DEBUG_LOG << "\nKV cache contains " << out_of_range_zp_count
                          << " ZP values outside typical [0,255] range (NOTE: this is valid for shifted data distributions):" << std::endl;
            for (const auto& [seq, tok, abs_pos, h, scale, zp] : out_of_range_zp_locations) {
                GPU_DEBUG_LOG << "  seq=" << seq << " token=" << tok << " (absolute_pos=" << abs_pos << ") head=" << h << " scale=" << scale << " zp=" << zp
                              << std::endl;
            }
            if (out_of_range_zp_locations.size() < static_cast<size_t>(out_of_range_zp_count)) {
                GPU_DEBUG_LOG << "  ... and " << (out_of_range_zp_count - out_of_range_zp_locations.size()) << " more" << std::endl;
            }
        }

        if (nan_count == 0 && inf_count == 0 && zero_scale_count == 0 && out_of_range_zp_count == 0) {
            GPU_DEBUG_LOG << "\nKV cache verification PASSED: " << total_tokens << " tokens checked";
            if (p.kv_cache_compression) {
                GPU_DEBUG_LOG << " (compression mode: " << (is_by_token ? "BY_TOKEN" : (is_by_channel ? "BY_CHANNEL" : "UNKNOWN")) << ")";
            } else {
                GPU_DEBUG_LOG << " (FP16 mode)";
            }
            GPU_DEBUG_LOG << std::endl;
        }
    }

private:
    // Helper structure to hold XAttention error statistics
    struct XAttentionErrorStats {
        int mismatch_count = 0;
        int catastrophic_count = 0;  // NaN, Inf, or error > 1.0
        int large_error_count = 0;   // error > 0.1
        float max_error = 0.0f;
        float avg_error = 0.0f;
        float avg_mismatch_error = 0.0f;
        size_t first_mismatch_idx = 0;
        float first_mismatch_actual = 0.0f;
        float first_mismatch_expected = 0.0f;
        bool found_first = false;

        // Separate counters for different abnormal value types
        int actual_nan_count = 0;
        int expected_nan_count = 0;
        int actual_inf_count = 0;
        int expected_inf_count = 0;
        int actual_gt1_count = 0;
        int expected_gt1_count = 0;
        int error_gt1_count = 0;

        // Track coordinates of abnormal values
        std::vector<std::tuple<size_t, size_t, size_t, float>> actual_nan_coords;  // (token, head, dim, value)
        std::vector<std::tuple<size_t, size_t, size_t, float>> expected_nan_coords;
        std::vector<std::tuple<size_t, size_t, size_t, float>> actual_inf_coords;
        std::vector<std::tuple<size_t, size_t, size_t, float>> expected_inf_coords;
        std::vector<std::tuple<size_t, size_t, size_t, float>> actual_gt1_coords;
        std::vector<std::tuple<size_t, size_t, size_t, float>> expected_gt1_coords;
    };

    // Collect XAttention error statistics by comparing GPU output with CPU reference
    static void collect_xattention_error_stats(const ov::float16* mem_ptr,
                                               const std::vector<ov::float16>& ref_output,
                                               size_t num_heads,
                                               size_t head_size,
                                               float tolerance,
                                               XAttentionErrorStats& stats) {
        size_t total_elements = ref_output.size();
        for (size_t i = 0; i < total_elements; i++) {
            float actual = static_cast<float>(mem_ptr[i]);
            float expected = static_cast<float>(ref_output[i]);
            float error = std::fabs(actual - expected);

            stats.avg_error += error;
            stats.max_error = std::max(stats.max_error, error);

            // Calculate coordinates: output layout is [T, Q * H]
            size_t head_dim_flat = i % (num_heads * head_size);
            size_t token_idx = i / (num_heads * head_size);
            size_t head_idx = head_dim_flat / head_size;
            size_t dim_idx = head_dim_flat % head_size;

            // Separate checks for actual vs expected with coordinate tracking
            if (std::isnan(actual)) {
                stats.actual_nan_count++;
                stats.actual_nan_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, actual));
            }
            if (std::isnan(expected)) {
                stats.expected_nan_count++;
                stats.expected_nan_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, expected));
            }
            if (std::isinf(actual)) {
                stats.actual_inf_count++;
                stats.actual_inf_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, actual));
            }
            if (std::isinf(expected)) {
                stats.expected_inf_count++;
                stats.expected_inf_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, expected));
            }
            if (!std::isnan(actual) && !std::isinf(actual) && std::fabs(actual) > 1.0f) {
                stats.actual_gt1_count++;
                stats.actual_gt1_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, actual));
            }
            if (!std::isnan(expected) && !std::isinf(expected) && std::fabs(expected) > 1.0f) {
                stats.expected_gt1_count++;
                stats.expected_gt1_coords.push_back(std::make_tuple(token_idx, head_idx, dim_idx, expected));
            }
            if (error > 1.0f)
                stats.error_gt1_count++;

            if (std::isnan(actual) || std::isinf(actual) || error > 1.0f) {
                stats.catastrophic_count++;
            }
            if (error > 0.1f) {
                stats.large_error_count++;
            }

            if (error > tolerance) {
                if (!stats.found_first) {
                    stats.first_mismatch_idx = i;
                    stats.first_mismatch_actual = actual;
                    stats.first_mismatch_expected = expected;
                    stats.found_first = true;
                }
                stats.avg_mismatch_error += error;
                stats.mismatch_count++;
            }
        }

        stats.avg_error /= total_elements;
        if (stats.mismatch_count > 0) {
            stats.avg_mismatch_error /= stats.mismatch_count;
        }
    }

    // Print coordinates of abnormal values (NaN/INF/large values)
    static void print_abnormal_value_coords(const std::string& label, const std::vector<std::tuple<size_t, size_t, size_t, float>>& coords, int count) {
        GPU_DEBUG_LOG << "    " << label << ": " << count << std::endl;
        if (count > 0) {
            GPU_DEBUG_LOG << "      Coordinates (token, head, dim, value):" << std::endl;
            size_t show_count = std::min(coords.size(), size_t(10));
            for (size_t j = 0; j < show_count; j++) {
                auto [t, h, d, v] = coords[j];
                GPU_DEBUG_LOG << "        [" << t << ", " << h << ", " << d << "] = " << v << std::endl;
            }
            if (coords.size() > 10) {
                GPU_DEBUG_LOG << "        ... and " << (coords.size() - 10) << " more" << std::endl;
            }
        }
    }

    // Print catastrophic error locations with detailed tags
    static void print_catastrophic_errors(const ov::float16* mem_ptr,
                                          const std::vector<ov::float16>& ref_output,
                                          size_t num_heads,
                                          size_t head_size,
                                          int catastrophic_count) {
        if (catastrophic_count > 0 && catastrophic_count <= 20) {
            GPU_DEBUG_LOG << "\nCatastrophic error locations (NaN/Inf/error>1.0):" << std::endl;
            size_t total_elements = ref_output.size();
            size_t total_tokens = total_elements / (num_heads * head_size);

            GPU_DEBUG_LOG << "  Dimensions: tokens=" << total_tokens << " heads=" << num_heads << " head_size=" << head_size << " (total=" << total_elements
                          << ")" << std::endl;

            for (size_t i = 0; i < total_elements; i++) {
                float actual = static_cast<float>(mem_ptr[i]);
                float expected = static_cast<float>(ref_output[i]);
                float error = std::fabs(actual - expected);
                if (std::isnan(actual) || std::isnan(expected) || std::isinf(actual) || std::isinf(expected) || error > 1.0f) {
                    // Memory layout: [token][head][dim] with innermost dimension first
                    size_t elements_per_token = num_heads * head_size;
                    size_t token = i / elements_per_token;
                    size_t within_token = i % elements_per_token;
                    size_t head = within_token / head_size;
                    size_t dim = within_token % head_size;

                    GPU_DEBUG_LOG << "  Index[" << i << "]: token=" << token << " head=" << head << " dim=" << dim << " | actual=" << actual
                                  << " expected=" << expected << " error=" << error;

                    // Tag each type
                    std::vector<std::string> tags;
                    if (std::isnan(actual))
                        tags.push_back("ACTUAL_NaN");
                    if (std::isnan(expected))
                        tags.push_back("EXPECTED_NaN");
                    if (std::isinf(actual))
                        tags.push_back("ACTUAL_INF");
                    if (std::isinf(expected))
                        tags.push_back("EXPECTED_INF");
                    if (!std::isnan(actual) && !std::isinf(actual) && std::fabs(actual) > 1.0f)
                        tags.push_back("ACTUAL_>1.0");
                    if (!std::isnan(expected) && !std::isinf(expected) && std::fabs(expected) > 1.0f)
                        tags.push_back("EXPECTED_>1.0");
                    if (error > 1.0f)
                        tags.push_back("ERROR_>1.0");

                    if (!tags.empty()) {
                        GPU_DEBUG_LOG << " [";
                        for (size_t t = 0; t < tags.size(); t++) {
                            if (t > 0) {
                                GPU_DEBUG_LOG << ", ";
                            }
                            GPU_DEBUG_LOG << tags[t];
                        }
                        GPU_DEBUG_LOG << "]";
                    }
                    GPU_DEBUG_LOG << std::endl;
                }
            }
        }
    }

    // Print error distribution by 1024-element blocks
    static void print_error_distribution(const ov::float16* mem_ptr, const std::vector<ov::float16>& ref_output, float tolerance) {
        GPU_DEBUG_LOG << "\nError distribution (by 1024-element blocks):" << std::endl;
        size_t block_size = 1024;
        size_t total_elements = ref_output.size();
        for (size_t block = 0; block < (total_elements + block_size - 1) / block_size && block < 10; block++) {
            int block_mismatches = 0;
            size_t start = block * block_size;
            size_t end = std::min(start + block_size, total_elements);
            for (size_t i = start; i < end; i++) {
                float error = std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(ref_output[i]));
                if (error > tolerance)
                    block_mismatches++;
            }
            GPU_DEBUG_LOG << "  Block " << block << " [" << start << "-" << end << "): " << block_mismatches << "/" << (end - start) << " ("
                          << (100.0 * block_mismatches / (end - start)) << "%)" << std::endl;
        }
    }

    // Print detailed XAttention error analysis
    static void print_xattention_error_details(const XAttentionErrorStats& stats,
                                               const ov::float16* mem_ptr,
                                               const std::vector<ov::float16>& ref_output,
                                               size_t num_heads,
                                               size_t head_size,
                                               float tolerance) {
        auto& engine = tests::get_test_engine();
        auto arch = engine.get_device_info().arch;
        std::string arch_name = (arch == cldnn::gpu_arch::xe2) ? "Xe2" : (arch == cldnn::gpu_arch::xe3) ? "Xe3" : "Xe1";
        size_t total_elements = ref_output.size();
        int allowed_mismatches = int(total_elements * 0.04);

        GPU_DEBUG_LOG << "\n=== XAttention Data Comparison Failed ===" << std::endl;
        GPU_DEBUG_LOG << "GPU Architecture: " << arch_name << " (arch=" << static_cast<int>(arch) << ")" << std::endl;
        GPU_DEBUG_LOG << "Total elements: " << total_elements << std::endl;
        GPU_DEBUG_LOG << "\nError Summary:" << std::endl;
        GPU_DEBUG_LOG << "  Mismatches (> tolerance): " << stats.mismatch_count << " (" << (100.0 * stats.mismatch_count / total_elements) << "%)" << std::endl;
        GPU_DEBUG_LOG << "  Allowed mismatches: " << allowed_mismatches << " (4%)" << std::endl;
        GPU_DEBUG_LOG << "  Large errors (> 0.1): " << stats.large_error_count << " (" << (100.0 * stats.large_error_count / total_elements) << "%)"
                      << std::endl;
        GPU_DEBUG_LOG << "  Catastrophic (NaN/Inf/>1.0): " << stats.catastrophic_count << std::endl;

        GPU_DEBUG_LOG << "\nAbnormal Value Analysis:" << std::endl;
        GPU_DEBUG_LOG << "  Actual output (GPU):" << std::endl;
        print_abnormal_value_coords("NaN values", stats.actual_nan_coords, stats.actual_nan_count);
        print_abnormal_value_coords("INF values", stats.actual_inf_coords, stats.actual_inf_count);
        print_abnormal_value_coords("Values > 1.0", stats.actual_gt1_coords, stats.actual_gt1_count);
        GPU_DEBUG_LOG << "  Expected output (CPU reference):" << std::endl;
        print_abnormal_value_coords("NaN values", stats.expected_nan_coords, stats.expected_nan_count);
        print_abnormal_value_coords("INF values", stats.expected_inf_coords, stats.expected_inf_count);
        print_abnormal_value_coords("Values > 1.0", stats.expected_gt1_coords, stats.expected_gt1_count);
        GPU_DEBUG_LOG << "  Error magnitude:" << std::endl;
        GPU_DEBUG_LOG << "    Errors > 1.0: " << stats.error_gt1_count << std::endl;
        GPU_DEBUG_LOG << "\nError Magnitudes:" << std::endl;
        GPU_DEBUG_LOG << "  Tolerance threshold: " << tolerance << std::endl;
        GPU_DEBUG_LOG << "  Max error: " << stats.max_error << std::endl;
        GPU_DEBUG_LOG << "  Avg error (all): " << stats.avg_error << std::endl;
        GPU_DEBUG_LOG << "  Avg error (mismatches only): " << stats.avg_mismatch_error << std::endl;
        GPU_DEBUG_LOG << "First mismatch at index " << stats.first_mismatch_idx << ":" << std::endl;
        GPU_DEBUG_LOG << "  Actual: " << stats.first_mismatch_actual << std::endl;
        GPU_DEBUG_LOG << "  Expected: " << stats.first_mismatch_expected << std::endl;
        GPU_DEBUG_LOG << "  Error: " << std::fabs(stats.first_mismatch_actual - stats.first_mismatch_expected) << std::endl;

        // Sample first 10 mismatches
        GPU_DEBUG_LOG << "\nFirst 10 mismatches:" << std::endl;
        int sample_count = 0;
        for (size_t i = 0; i < total_elements && sample_count < 10; i++) {
            float actual = static_cast<float>(mem_ptr[i]);
            float expected = static_cast<float>(ref_output[i]);
            float error = std::fabs(actual - expected);
            if (error > tolerance) {
                GPU_DEBUG_LOG << "  [" << i << "] actual=" << actual << " expected=" << expected << " error=" << error << std::endl;
                sample_count++;
            }
        }

        print_catastrophic_errors(mem_ptr, ref_output, num_heads, head_size, stats.catastrophic_count);

        // Analyze error type
        GPU_DEBUG_LOG << "\nError Analysis:" << std::endl;
        if (stats.catastrophic_count > 0) {
            GPU_DEBUG_LOG << "  ERROR TYPE: CATASTROPHIC - Contains NaN/Inf or very large errors" << std::endl;
            GPU_DEBUG_LOG << "  This indicates a serious bug, not just precision issues." << std::endl;
        } else if (stats.large_error_count > stats.mismatch_count * 0.5) {
            GPU_DEBUG_LOG << "  ERROR TYPE: LARGE - Most mismatches have error > 0.1" << std::endl;
            GPU_DEBUG_LOG << "  This suggests wrong calculations, not precision drift." << std::endl;
        } else {
            GPU_DEBUG_LOG << "  ERROR TYPE: PRECISION - Most errors are small" << std::endl;
            GPU_DEBUG_LOG << "  This suggests accumulated floating-point errors." << std::endl;
        }

        // Comment on tolerance metric
        GPU_DEBUG_LOG << "\nTolerance Metric Analysis:" << std::endl;
        GPU_DEBUG_LOG << "  Using: Absolute error with threshold " << tolerance << std::endl;
        GPU_DEBUG_LOG << "  Input range: Normal(0, 0.1), typically [-0.3, 0.3]" << std::endl;
        GPU_DEBUG_LOG << "  Relative error at typical values: ~" << (tolerance / 0.1 * 100) << "%" << std::endl;
        GPU_DEBUG_LOG << "  Verdict: Tolerance is reasonable for FP16 attention calculations" << std::endl;

        print_error_distribution(mem_ptr, ref_output, tolerance);
        GPU_DEBUG_LOG << "========================================\n" << std::endl;
    }

public:
    static bool check_cm_available() {
        auto& engine = tests::get_test_engine();
        ov::intel_gpu::ExecutionConfig config = tests::get_test_default_config(engine);
        return cldnn::check_cm_jit_support(engine, config) && engine.get_device_info().supports_immad;
    }
};

struct paged_attention_test_params {
    std::vector<SubsequenceDescriptor> subsequences;
    int num_heads;
    int num_kv_heads;
    int k_head_size;
    int v_head_size;
    int block_size;
    int sliding_window_size;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    bool dynamic_paddings;
    ScoresMode scores_mode;
    CacheRotationDescriptor rotation_config;
    bool disable_flashattn_v2;
    bool has_adaptive_rkv = false;
    int start_size = 0;                // Common start_size for all sequences
    std::vector<int> evictable_sizes;  // Per-sequence evictable sizes

    // XAttention-related params are grouped below.
    bool has_xattention = false;
    std::optional<std::vector<float>> xattention_threshold = std::nullopt;
    std::optional<std::vector<int>> xattention_block_size = std::nullopt;

    ov::element::Type kv_cache_precision = ov::element::dynamic;

    // test query-to-query attention bias
    bool has_qq_bias = false;
    QueryToQueryAttentionDescriptor qq_bias_config = {};
    bool run_reference = true;

    // optional token_type_ids passed to PagedAttention; if set (non-empty), it is forwarded
    // to the op as the TOKEN_TYPE_IDS input. When std::nullopt, a default {0} buffer is used.
    std::optional<std::vector<int>> token_type_ids = std::nullopt;

    // Sink input testing: when true, enables has_sink_input on the primitive and forces
    // the sdpa_opt.cl path (by setting has_token_type_ids=true to disable micro-SDPA).
    bool has_sink_input = false;
    // When set, overrides the default sink values in PAM before memory allocation.
    std::optional<std::vector<ov::float16>> sink_values = std::nullopt;
    // When true, forces could_use_flashattn_v2(true) to guarantee the FA_V2 kernel path.
    bool force_flashattn_v2 = false;
};

const auto ENABLE_CACHE_COMPRESSION = true;
const auto DISABLE_CACHE_COMPRESSION = false;
const auto DISABLE_SCORES = ScoresMode::DISABLED;
const auto ENABLE_SCORES = ScoresMode::LAST_TOKEN;
const auto ENABLE_SCORES_SNAPKV = ScoresMode::SNAPKV;
const auto PER_BLOCK_ROTATION = CacheRotationDescriptor{true, true};
const auto PER_TOKEN_ROTATION = CacheRotationDescriptor{true, false};
const auto DISABLE_ROTATION = CacheRotationDescriptor{false, false};
const auto STATIC_INPUT_PAD = false;
const auto DYNAMIC_INPUT_PAD = true;
const auto ENABLE_FA_V2 = false;
const auto DISABLE_FA_V2 = true;
const auto ENABLE_DIVERSITY = true;
const auto DISABLE_DIVERSITY = false;
const auto ENABLE_QQ_BIAS = QueryToQueryAttentionDescriptor{{{{1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1}}}, {0, 16}};
