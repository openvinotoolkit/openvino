// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include <openvino/reference/xattention.hpp>
#include <openvino/reference/adaptive_rkv_diversity.hpp>

#include "openvino/runtime/tensor.hpp"
#include "primitive_inst.h"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

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
 */

enum class ScoresMode {
    DISABLED = 0,
    LAST_TOKEN,
    SNAPKV
};

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

struct PagedAttentionManager {
    int num_heads;
    int num_kv_heads;
    int k_head_size;
    int v_head_size;
    int block_size;
    int sliding_window_size;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    bool has_score_aggregation;
    bool has_xattention;
    CacheRotationDescriptor rotation_config;
    std::vector<SubsequenceDescriptor> subsequence_descs;

    // per-subsequence QKV inputs
    std::vector<std::vector<ov::float16>> query_data; // {[1, num_tokens, num_heads, k_head_size], ..}
    std::vector<std::vector<ov::float16>> key_data;   // {[1, past_len + num_tokens, num_heads, k_head_size], ..}
    std::vector<std::vector<ov::float16>> value_data; // {[1, past_len + num_tokens, num_heads, v_head_size], ..}

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

    std::vector<ov::float16> xattention_threshold;
    std::vector<int> xattention_block_size;
    std::vector<int> xattention_stride;

    std::vector<ov::float16> sinks;

    int adaptive_rkv_start_size = 0;
    std::vector<int> adaptive_rkv_evictable_sizes;
    std::vector<int> adaptive_rkv_diversity_block_set_indices;
    std::vector<int> adaptive_rkv_diversity_block_set_indices_begins;

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
                          std::vector<float> threshold)
        : num_heads(num_heads)
        , num_kv_heads(num_kv_heads)
        , k_head_size(k_head_size)
        , v_head_size(v_head_size)
        , block_size(block_size)
        , sliding_window_size(sliding_window_size)
        , kv_cache_compression(kv_cache_compression)
        , key_cache_quant_mode(key_cache_quant_mode)
        , has_score_aggregation(has_score_aggregation)
        , has_xattention(has_xattention)
        , rotation_config(rotation_config)
        , subsequence_descs(subsequence_descs)
        , test_engine(engine)
        , test_stream(stream)
        , rg(rg) {
        // init subsequence_begins and block_indices_begins
        subsequence_begins.push_back(0);
        block_indices_begins.push_back(0);
        for (int i = 0; i < static_cast<int>(threshold.size()); i++) {
            xattention_threshold.emplace_back(static_cast<ov::float16>(threshold[i]));
        }

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
            int required_blocks = ceil_div(subsequence_length, block_size);
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
                rotation_deltas = generate_rotation_deltas_data(rg,
                                                                max_context_len[0],
                                                                rotated_block_indices.size(),
                                                                block_size,
                                                                rotation_config.per_block);
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

    memory::ptr get_query_memory() {
        return get_QKV_memory(query_data, num_heads, k_head_size, false);
    }

    memory::ptr get_key_memory() {
        return get_QKV_memory(key_data, num_kv_heads, k_head_size, true);
    }

    memory::ptr get_value_memory() {
        return get_QKV_memory(value_data, num_kv_heads, v_head_size, true);
    }

    memory::ptr get_key_cache_memory_cm() {
        auto key_cache_dt = kv_cache_compression ? data_types::i8 : data_types::f16;
        const int head_size = k_head_size;
        const int adjusted_head_size = head_size + (kv_cache_compression ? 4 : 0);

        const auto num_blocks = block_indices.back() + 1;
        auto key_cache_shape = ov::PartialShape{static_cast<int64_t>(num_blocks),
                                                static_cast<int64_t>(num_kv_heads),
                                                static_cast<int64_t>(block_size),
                                                static_cast<int64_t>(adjusted_head_size)};
        auto key_cache_layout = layout{key_cache_shape, key_cache_dt, format::bfyx};
        auto memory = test_engine.allocate_memory(key_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const int past_len = subsequence_descs[i].past_len;
            if (past_len == 0)
                continue;

            const int blocks_num = ceil_div(past_len + 1, block_size);
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
                        } else {
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
                        }
                    }
                }
            }
        }
        return memory;
    }

    memory::ptr get_key_cache_memory() {
        auto key_cache_dt = data_types::f16;
        auto adjusted_head_size = k_head_size;
        auto adjusted_block_size = block_size;
        if (kv_cache_compression) {
            key_cache_dt = data_types::i8;
            if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                adjusted_block_size += 4;
            } else {
                adjusted_head_size += 4;
            }
        }

        auto num_blocks = block_indices.back() + 1;
        auto key_cache_shape = ov::PartialShape{ num_blocks, num_kv_heads, adjusted_head_size, adjusted_block_size };
        auto key_cache_layout = layout{ key_cache_shape, key_cache_dt, format::bfyx };
        auto memory = test_engine.allocate_memory(key_cache_layout);
        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            int past_len = subsequence_descs[i].past_len;
            if (past_len != 0) {
                int blocks_num = ceil_div(past_len + 1, block_size);
                int start_block_idx = block_indices[block_indices_begins[i]];
                for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                    int last_token_idx = block_idx == blocks_num - 1 ? (past_len - block_size * block_idx)
                                                                     : block_size;
                    // quantize by channel
                    if (kv_cache_compression && key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                        for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                            for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                std::vector<ov::float16> token_block(block_size);
                                for (int token_idx = 0; token_idx < last_token_idx; ++token_idx) {
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    token_block[token_idx] = *(key_data[i].data() + input_token_offset * num_kv_heads * k_head_size + head_idx * k_head_size + k_head_size_idx);
                                }
                                auto [quantized_data, scale, zp] = quantize_data(token_block.data(), last_token_idx, true);
                                size_t output_block_offset = (start_block_idx + block_idx) * num_kv_heads * adjusted_head_size * adjusted_block_size +
                                                             head_idx * adjusted_head_size * adjusted_block_size;
                                size_t output_offset = output_block_offset +
                                                       k_head_size_idx * adjusted_block_size;
                                set_values(test_stream, memory, quantized_data.data(), last_token_idx, output_offset);
                                size_t comp_offset = (output_offset + block_size)/2;
                                set_values(test_stream, memory, &scale, 1, comp_offset);
                                set_values(test_stream, memory, &zp, 1, comp_offset + 1);
                            }
                        }
                    }
                    for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                        for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
                            if (kv_cache_compression) {
                                if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_TOKEN) {
                                    // quantize by token
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    ov::float16* data_ptr = key_data[i].data() +
                                                            input_token_offset * num_kv_heads * k_head_size +
                                                            head_idx * k_head_size;
                                    // shape: [num_blocks, num_kv_heads, adjusted_head_size, block_size]
                                    size_t output_block_offset = (start_block_idx + block_idx) * num_kv_heads * adjusted_head_size * block_size +
                                                                 head_idx * adjusted_head_size * block_size;

                                    auto [quantized_data, scale, zp] = quantize_data(data_ptr, k_head_size);
                                    for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                        auto quantized_data_ptr = quantized_data.data() + k_head_size_idx;

                                        size_t output_offset = output_block_offset +
                                                               k_head_size_idx * block_size +
                                                               token_idx;

                                        set_values(test_stream, memory, quantized_data_ptr, 1, output_offset);
                                    }
                                    size_t comp_offset = (output_block_offset + k_head_size * block_size) / 2;
                                    set_values(test_stream, memory, &scale, 1, comp_offset + token_idx);
                                    set_values(test_stream, memory, &zp, 1, comp_offset + block_size + token_idx);
                                }
                            } else {
                                for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    ov::float16* data_ptr = key_data[i].data() +
                                                            input_token_offset * num_kv_heads * k_head_size +
                                                            head_idx * k_head_size + k_head_size_idx;

                                    // shape: [num_blocks, num_kv_heads, k_head_size, block_size]
                                    size_t output_offset = (start_block_idx + block_idx) * num_kv_heads * k_head_size * block_size +
                                                           head_idx * k_head_size * block_size +
                                                           k_head_size_idx * block_size +
                                                           token_idx;

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

    memory::ptr get_value_cache_memory() {
        auto value_cache_dt = kv_cache_compression ? data_types::i8 : data_types::f16;
        const int head_size = v_head_size;

        const int adjusted_head_size = head_size + (kv_cache_compression ? 4 : 0);

        const auto num_blocks = block_indices.back() + 1;
        auto value_cache_shape = ov::PartialShape{static_cast<int64_t>(num_blocks),
                                                  static_cast<int64_t>(num_kv_heads),
                                                  static_cast<int64_t>(block_size),
                                                  static_cast<int64_t>(adjusted_head_size)};
        auto value_cache_layout = layout{value_cache_shape, value_cache_dt, format::bfyx};
        auto memory = test_engine.allocate_memory(value_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const int past_len = subsequence_descs[i].past_len;
            if (past_len == 0)
                continue;

            const int blocks_num = ceil_div(past_len + 1, block_size);
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

    memory::ptr get_past_lens_memory() {
        return get_memory_from_vec(past_lens);
    }

    memory::ptr get_subsequence_begins_memory() {
        return get_memory_from_vec(subsequence_begins);
    }

    memory::ptr get_block_indices_memory() {
        return get_memory_from_vec(block_indices);
    }

    memory::ptr get_block_indices_begins_memory() {
        return get_memory_from_vec(block_indices_begins);
    }

    memory::ptr get_scale_memory() {
        std::vector<ov::float16> scale = { ov::float16(get_default_scale()) };
        return get_memory_from_vec(scale);
    }

    memory::ptr get_sliding_window_memory() {
        std::vector<int> sliding_window = { 0 };
        return get_memory_from_vec(sliding_window);
    }

    memory::ptr get_alibi_memory() {
        std::vector<ov::float16> alibi;
        return get_memory_from_vec(alibi);
    }

    memory::ptr get_max_context_len_memory() {
        return get_memory_from_vec(max_context_len);
    }

    memory::ptr get_score_aggregation() {
        return get_memory_from_vec(score_aggregation);
    }

    memory::ptr get_rotated_block_indices_memory() {
        return get_memory_from_vec(rotated_block_indices);
    }

    memory::ptr get_rotation_deltas_memory() {
        auto mem = get_memory_from_vec(rotation_deltas);
        auto layout = mem->get_layout();
        auto last_dim = rotation_config.per_block ? 1 : block_size;
        layout.set_partial_shape(ov::PartialShape{ static_cast<long int>(rotated_block_indices.size()), last_dim });

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    memory::ptr get_rotation_trig_lut_memory() {
        auto mem = get_memory_from_vec(rotation_trig_lut);
        auto layout = mem->get_layout();
        layout.set_partial_shape(ov::PartialShape{ max_context_len[0], k_head_size });

        if (rotated_block_indices.empty()) {
            auto empty_layout = mem->get_layout();
            empty_layout.set_partial_shape(ov::PartialShape{ 0, k_head_size });
            return test_engine.reinterpret_buffer(*mem, empty_layout);
        }

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    memory::ptr get_xattention_threshold_memory() {
        auto mem = get_memory_from_vec(xattention_threshold);
        auto layout = mem->get_layout();
        layout.set_partial_shape(ov::PartialShape{ 1 });

        if (xattention_threshold.empty()) {
            auto empty_layout = mem->get_layout();
            empty_layout.set_partial_shape(ov::PartialShape{ 0 });
            return test_engine.reinterpret_buffer(*mem, empty_layout);
        }

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    memory::ptr get_xattention_block_size_memory() {
        return get_memory_from_vec(xattention_block_size);
    }

    memory::ptr get_xattention_stride_memory() {
        return get_memory_from_vec(xattention_stride);
    }

    memory::ptr get_sinks_memory() {
        auto mem = get_memory_from_vec(sinks);
        auto layout = mem->get_layout();
        layout.set_partial_shape(ov::PartialShape{ 1, num_heads, 1, 1 });

        if (sinks.empty()) {
            auto empty_layout = mem->get_layout();
            empty_layout.set_partial_shape(ov::PartialShape{ 0, 0, 0, 0 });
            return test_engine.reinterpret_buffer(*mem, empty_layout);
        }

        return test_engine.reinterpret_buffer(*mem, layout);
    }

    memory::ptr get_adaptive_rkv_start_size_memory() {
        auto mem = test_engine.allocate_memory({{}, data_types::i32, format::bfyx});
        mem_lock<int> lock(mem, test_stream);
        lock[0] = adaptive_rkv_start_size;
        return mem;
    }

    memory::ptr get_adaptive_rkv_evictable_sizes_memory() {
        return get_memory_from_vec(adaptive_rkv_evictable_sizes);
    }

    memory::ptr get_adaptive_rkv_diversity_block_set_indices_memory() {
        return get_memory_from_vec(adaptive_rkv_diversity_block_set_indices);
    }

    memory::ptr get_adaptive_rkv_diversity_block_set_indices_begins_memory() {
        return get_memory_from_vec(adaptive_rkv_diversity_block_set_indices_begins);
    }

    float get_default_scale() {
        return static_cast<float>(1.f / std::sqrt(k_head_size));
    }

private:
    template<typename T>
    memory::ptr get_memory_from_vec(std::vector<T>& input_data) {
        auto data_size = input_data.empty() ? 1 : input_data.size();
        auto shape = ov::PartialShape{ static_cast<int>(data_size) };
        auto layout = cldnn::layout{ shape, ov::element::from<T>(), format::bfyx };
        auto memory = test_engine.allocate_memory(layout);

        if (input_data.empty()) {
            auto shape = ov::PartialShape{0};
            auto layout = cldnn::layout{ shape, ov::element::from<T>(), format::bfyx };
            return test_engine.reinterpret_buffer(*memory, layout);
        }

        set_values(test_stream, memory, input_data.data(), input_data.size(), 0);

        return memory;
    }

    memory::ptr get_QKV_memory(std::vector<std::vector<ov::float16>>& input_data, int num_heads, int head_size, bool skip_past_len) {
        int total_tokens = 0;
        for (const auto& subsequence_desc : subsequence_descs)
            total_tokens += subsequence_desc.num_tokens;

        auto query_shape = ov::PartialShape{ total_tokens, num_heads * head_size };
        auto query_layout = layout{ query_shape, data_types::f16, format::bfyx };
        auto memory = test_engine.allocate_memory(query_layout);

        for (int subsequence_idx = 0; subsequence_idx < static_cast<int>(subsequence_descs.size()); subsequence_idx++) {
            for (int token_idx = 0; token_idx < subsequence_descs[subsequence_idx].num_tokens; token_idx++) {
                for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                    size_t input_token_offset = token_idx;
                    // as generated data stored in vectors includes past_len, ignore it for KV inputs
                    if (skip_past_len)
                        input_token_offset += subsequence_descs[subsequence_idx].past_len;

                    ov::float16* data_ptr = input_data[subsequence_idx].data() +
                                            input_token_offset * num_heads * head_size +
                                            head_idx * head_size;

                    size_t output_token_offset = subsequence_begins[subsequence_idx] + token_idx;
                    size_t output_offset = output_token_offset * num_heads * head_size +
                                           head_idx * head_size;

                    set_values(test_stream, memory, data_ptr, head_size, output_offset);
                }
            }
        }

        return memory;
    }

    template<typename T>
    static void set_values(stream& stream, memory::ptr mem, T* vals, size_t size, size_t dst_offset) {
        mem_lock<T> mem_ptr(mem, stream);
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

    static std::vector<int> generate_rotation_deltas_data(tests::random_generator& rg, size_t max_tokens_num, size_t rotated_blocks_num, size_t block_size, bool per_block) {
        const size_t total_elements_num = per_block ? rotated_blocks_num
                                                    : rotated_blocks_num * block_size;
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
    uint64_t operator()(const ov::float16 __val) const {
        return std::hash<float>()(__val);
    }
};
}

struct PagedAttentionReference {
    PagedAttentionReference(PagedAttentionManager& pam)
    : pam(pam)
    , test_engine(pam.test_engine)
    , test_stream(pam.test_stream) {}

    std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> get_reference(bool has_xattention, std::vector<float> threshold, memory::ptr key_cache_mem = nullptr) {
        std::vector<ov::float16> ref_data_output;
        std::vector<ov::float16> ref_scores_output;
        std::vector<ov::float16> ref_diversity_output;

        for (size_t i = 0; i < pam.subsequence_descs.size(); i++) {
            const auto& subsequence_desc = pam.subsequence_descs[i];
            const auto kv_seq_len = subsequence_desc.num_tokens + subsequence_desc.past_len;

            auto key_data = pam.key_data[i];
            if (pam.rotation_config.apply_rotation) {
                auto blocks_start = pam.block_indices_begins[i];
                auto blocks_end = pam.block_indices_begins[i + 1];

                std::vector<int> block_indices(pam.block_indices.begin() + blocks_start,
                                               pam.block_indices.begin() + blocks_end);

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
            double th = static_cast<double>(threshold.size() == 1 ? threshold[0] : threshold[i]);
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
                                                         th);

            // concatenate all subsequences into one vector
            ref_data_output.insert(ref_data_output.end(),
                                   subsequence_ref_results.first.begin(),
                                   subsequence_ref_results.first.end());
            ref_scores_output.insert(ref_scores_output.end(),
                                     subsequence_ref_results.second.begin(),
                                     subsequence_ref_results.second.end());
        }

        if (!pam.adaptive_rkv_evictable_sizes.empty()) {
            ref_diversity_output = compute_diversity_reference(key_cache_mem);
        }

        return { ref_data_output, ref_scores_output, ref_diversity_output };
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
                                                                                double threshold = 0.9,
                                                                                size_t block_size = 128,
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
                        size_t src_key_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(src_kv_head)) * static_cast<size_t>(k_head_size);
                        size_t dst_key_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) * static_cast<size_t>(k_head_size);
                        for (int d = 0; d < k_head_size; ++d)
                            expanded_key_data[dst_key_off + static_cast<size_t>(d)] = key_data[src_key_off + static_cast<size_t>(d)];
    
                        size_t src_val_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_kv_heads) + static_cast<size_t>(src_kv_head)) * static_cast<size_t>(v_head_size);
                        size_t dst_val_off = (static_cast<size_t>(key_idx) * static_cast<size_t>(num_heads) + static_cast<size_t>(h)) * static_cast<size_t>(v_head_size);
                        for (int d = 0; d < v_head_size; ++d)
                            expanded_value_data[dst_val_off + static_cast<size_t>(d)] = value_data[src_val_off + static_cast<size_t>(d)];
                    }
                }
    
                key_shape = ov::PartialShape{1, num_keys, num_heads, k_head_size};
                value_shape = ov::PartialShape{1, num_keys, num_heads, v_head_size};
                num_kv_heads = num_heads;
            }
        }

        auto query_layout = layout{query_shape, data_types::f16, format::bfyx};
        auto key_layout = layout{key_shape, data_types::f16, format::bfyx};
        auto value_layout = layout{value_shape, data_types::f16, format::bfyx};
        auto scale_layout = cldnn::layout({1}, data_types::f16, format::bfyx);

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

        set_values(query_mem, query_data);
        if (do_gqa_expand) {
            set_values(key_mem, expanded_key_data);
            set_values(value_mem, expanded_value_data);
        } else {
            set_values(key_mem, key_data);
            set_values(value_mem, value_data);
        }
        set_values(scale_mem, {static_cast<ov::float16>(scale)});

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
            ov::reference::XAttentionBlockSelector<float> selector(threshold, block_size, stride);
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
                                                         static_cast<int>(block_size));
        topology topology;
        if (num_heads == num_kv_heads) {
            topology.add(input_layout("query", query_layout),
                        input_layout("key", key_layout),
                        input_layout("value", value_layout),
                        data("mask", mask_mem),
                        data("scale", scale_mem),
                        permute("query_transposed", input_info("query"), {0, 2, 1, 3}),
                        permute("key_transposed", input_info("key"), {0, 2, 3, 1}),
                        permute("value_transposed", input_info("value"), {0, 2, 1, 3}),
                        gemm("qk_gemm", { input_info("query_transposed"), input_info("key_transposed") }, data_types::f16, false, false),
                        eltwise("scale_div", { input_info("qk_gemm"), input_info("scale") }, eltwise_mode::prod),
                        eltwise("eltwise", { input_info("scale_div"), input_info("mask") }, eltwise_mode::sum),
                        softmax("softmax", input_info("eltwise"), -1),
                        gemm("qkv_gemm", { input_info("softmax"), input_info("value_transposed") }, data_types::f16, false, false),
                        permute("qkv_gemm_transposed", input_info("qkv_gemm"), {0, 2, 1, 3}),
                        reorder("output_data", input_info("qkv_gemm_transposed"), format::bfyx, data_types::f16),
                        reorder("scores_data", input_info("softmax"), format::bfyx, data_types::f16)
            );
        } else {
            topology.add(input_layout("query", query_layout),
                        input_layout("key", key_layout),
                        input_layout("value", value_layout),
                        data("mask", mask_mem),
                        data("scale", scale_mem),
                        permute("query_transposed", input_info("query"), {1, 2, 0, 3}),
                        permute("key_transposed", input_info("key"), {1, 2, 3, 0}),
                        permute("value_transposed", input_info("value"), {1, 2, 0, 3}),
                        gemm("qk_gemm", { input_info("query_transposed"), input_info("key_transposed") }, data_types::f16, false, false),
                        eltwise("scale_div", { input_info("qk_gemm"), input_info("scale") }, eltwise_mode::prod),
                        eltwise("eltwise", { input_info("scale_div"), input_info("mask") }, eltwise_mode::sum),
                        softmax("softmax", input_info("eltwise"), -1),
                        gemm("qkv_gemm", { input_info("softmax"), input_info("value_transposed") }, data_types::f16, false, false),
                        reshape("qkv_gemm_reshape", input_info("qkv_gemm"), {1, num_heads, v_head_size, num_queries}),
                        permute("qkv_gemm_transposed", input_info("qkv_gemm_reshape"), {0, 2, 1, 3}),
                        reorder("output_data", input_info("qkv_gemm_transposed"), format::bfyx, data_types::f16),
                        reorder("scores_data", input_info("softmax"), format::bfyx, data_types::f16)
            );
        }

        ExecutionConfig config = get_test_default_config(test_engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network::ptr network = get_network(test_engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);

        auto outputs = network->execute();

        auto output_data_mem = outputs.at("output_data").get_memory();
        auto output_scores_mem = outputs.at("scores_data").get_memory();

        return { get_output_data_vec(output_data_mem, num_queries, v_head_size, num_heads),
                 get_output_scores_vec(output_scores_mem, window_size, num_queries, num_keys, num_heads) };
    }

    std::vector<ov::float16> get_output_scores_vec(memory::ptr scores_output,
                                                   int window_size,
                                                   int num_queries,
                                                   int num_keys,
                                                   int num_heads) {
        OPENVINO_ASSERT(scores_output->count() == static_cast<size_t>(num_heads * num_queries * num_keys));

        std::vector<ov::float16> output_scores(num_keys, 0);
        mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output, test_stream);
        for (int row_idx = 0; row_idx < window_size; row_idx++) {
            for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                for (int score_idx = 0; score_idx < num_keys; score_idx++) {
                    auto scores_offset = head_idx * num_queries * num_keys +
                                         (num_queries - window_size + row_idx) * num_keys +
                                         score_idx;
                    output_scores[score_idx] += mem_ptr[scores_offset];
                }
            }
        }

        return output_scores;
    }

    std::vector<ov::float16> get_output_data_vec(memory::ptr data_output,
                                                 int num_queries,
                                                 int k_head_size,
                                                 int num_heads) {
        OPENVINO_ASSERT(data_output->count() == static_cast<size_t>(num_queries * num_heads * k_head_size));

        std::vector<ov::float16> output_data(data_output->count());
        mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output, test_stream);
        for (size_t i = 0; i < data_output->count(); i++)
            output_data[i] = mem_ptr[i];

        return output_data;
    }

    memory::ptr get_mask_mem_combined_multi_head(int num_queries,
                                                 int num_keys,
                                                 int num_heads,
                                                 int num_kv_heads,
                                                 int sliding_window_size,
                                                 const ov::reference::XAttentionRetainedBlockIndicesForAllHeads& retained_blocks,
                                                 int block_size) {
        int heads_per_kv = num_heads / num_kv_heads;

        ov::PartialShape mask_shape;
        if (retained_blocks.empty()) {
            mask_shape = ov::PartialShape{1, 1, num_queries, num_keys};
        } else if (num_heads == num_kv_heads) {
            mask_shape = ov::PartialShape{1, num_heads, num_queries, num_keys};
        } else {
            mask_shape = ov::PartialShape{num_kv_heads, heads_per_kv, num_queries, num_keys};
        }

        auto mask_layout = layout{mask_shape, data_types::f16, format::bfyx};
        auto mask_mem = test_engine.allocate_memory(mask_layout);
        mem_lock<ov::float16> mem_ptr(mask_mem, test_stream);

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
                    auto input_offset = (start_token_idx + token_idx) * num_heads * k_head_size +
                                        head_idx * k_head_size +
                                        k_head_size_idx;

                    auto cache_value_0 = cache_data[input_offset];
                    auto cache_value_1 = cache_data[input_offset + k_head_size / 2];

                    ov::float16 rotation_value_cos = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx];
                    ov::float16 rotation_value_sin = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx + k_head_size / 2];

                    cache_data[input_offset]                 = cache_value_0 * rotation_value_cos - cache_value_1 * rotation_value_sin;
                    cache_data[input_offset + k_head_size / 2] = cache_value_0 * rotation_value_sin + cache_value_1 * rotation_value_cos;
                }
            }
        }
    }

    std::vector<ov::float16> read_key_from_cache(memory::ptr key_cache_mem, size_t seq_idx, int total_tokens) {
        // Read key vectors from key_cache memory
        // key_cache layout: [num_blocks, num_kv_heads, head_size, block_size]
        std::vector<ov::float16> key_data(pam.num_kv_heads * total_tokens * pam.k_head_size);

        const int blocks_start = pam.block_indices_begins[seq_idx];
        const int blocks_end = pam.block_indices_begins[seq_idx + 1];
        const int num_blocks = blocks_end - blocks_start;

        const bool is_compressed = pam.kv_cache_compression;

        if (!is_compressed) {
            // Uncompressed case: read as float16
            mem_lock<ov::float16, mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);

            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                const int physical_block = pam.block_indices[blocks_start + block_idx];
                const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                    const int token_idx = block_idx * pam.block_size + token_offset;

                    for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                        const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * pam.k_head_size * pam.block_size +
                                                  static_cast<size_t>(head_idx) * pam.k_head_size * pam.block_size;

                        const size_t output_base = static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size +
                                                   static_cast<size_t>(token_idx) * pam.k_head_size;

                        for (int dim = 0; dim < pam.k_head_size; dim++) {
                            const size_t cache_offset = cache_base + static_cast<size_t>(dim) * pam.block_size + token_offset;
                            key_data[output_base + dim] = cache_ptr[cache_offset];
                        }
                    }
                }
            }
        } else {
            // Compressed case: read as int8 and dequantize
            mem_lock<int8_t, mem_lock_type::read> cache_ptr(key_cache_mem, test_stream);

            if (pam.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                // BY_CHANNEL: [num_blocks, num_kv_heads, head_size, block_size+4]
                // Each dimension quantized across all tokens in block
                for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                    const int physical_block = pam.block_indices[blocks_start + block_idx];
                    const int tokens_in_block = std::min(pam.block_size, total_tokens - block_idx * pam.block_size);

                    for (int head_idx = 0; head_idx < pam.num_kv_heads; head_idx++) {
                        const size_t cache_base = static_cast<size_t>(physical_block) * pam.num_kv_heads * pam.k_head_size * (pam.block_size + 4) +
                                                  static_cast<size_t>(head_idx) * pam.k_head_size * (pam.block_size + 4);

                        for (int dim = 0; dim < pam.k_head_size; dim++) {
                            // Read scale and zero-point for this dimension
                            const size_t scale_offset = cache_base + static_cast<size_t>(dim) * (pam.block_size + 4) + pam.block_size;
                            ov::float16 scale = *reinterpret_cast<const ov::float16*>(&cache_ptr[scale_offset]);
                            ov::float16 zp = *reinterpret_cast<const ov::float16*>(&cache_ptr[scale_offset + 2]);

                            // Dequantize all tokens for this dimension
                            for (int token_offset = 0; token_offset < tokens_in_block; token_offset++) {
                                const int token_idx = block_idx * pam.block_size + token_offset;
                                const size_t cache_offset = cache_base + static_cast<size_t>(dim) * (pam.block_size + 4) + token_offset;
                                const size_t output_offset = static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size +
                                                             static_cast<size_t>(token_idx) * pam.k_head_size + dim;

                                int8_t quantized_value = cache_ptr[cache_offset];
                                float dequantized = (static_cast<float>(quantized_value) - static_cast<float>(zp)) * static_cast<float>(scale);
                                key_data[output_offset] = ov::float16(dequantized);
                            }
                        }
                    }
                }
            } else {
                // BY_TOKEN: [num_blocks, num_kv_heads, head_size+4, block_size]
                // Token-wise quantization with shared scale/zp per token
                // Layout: data rows [0..head_size-1], scale at [head_size], zp at [head_size+2] (fp16)
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

                            const size_t output_base = static_cast<size_t>(head_idx) * total_tokens * pam.k_head_size +
                                                       static_cast<size_t>(token_idx) * pam.k_head_size;

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
            }
        }
        return key_data;
    }

    std::vector<ov::float16> compute_diversity_reference(memory::ptr key_cache_mem) {
        std::vector<ov::float16> diversity_output;

        for (size_t seq_idx = 0; seq_idx < pam.subsequence_descs.size(); seq_idx++) {
            const auto start_size = pam.adaptive_rkv_start_size;
            const auto evictable_size = pam.adaptive_rkv_evictable_sizes[seq_idx];

            // Read key data from key_cache instead of original key_data
            const auto& subsequence_desc = pam.subsequence_descs[seq_idx];
            const auto total_tokens = subsequence_desc.num_tokens + subsequence_desc.past_len;

            // Extract key vectors from key_cache memory
            std::vector<ov::float16> key_data = read_key_from_cache(key_cache_mem, seq_idx, total_tokens);

            ov::Shape key_shape = {static_cast<size_t>(pam.num_kv_heads),
                                   static_cast<size_t>(total_tokens),
                                   static_cast<size_t>(pam.k_head_size)};

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
    random_generator rg;
    cldnn::engine& engine = get_test_engine();
    float tolerance = 2e-3;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void execute(T& p) {
        PagedAttentionManager pam(rg,
                                  get_test_engine(),
                                  get_test_stream(),
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
                                  p.threshold);

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

        if (p.kv_cache_compression)
            tolerance = 25e-3;

        auto query_mem = pam.get_query_memory();
        auto key_mem = pam.get_key_memory();
        auto value_mem = pam.get_value_memory();
        
        memory::ptr key_cache_mem;
        if (p.has_xattention) {
            key_cache_mem = pam.get_key_cache_memory_cm();
        } else {
            key_cache_mem = pam.get_key_cache_memory();
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

        auto query_layout = query_mem->get_layout();
        auto key_layout = key_mem->get_layout();
        auto value_layout = value_mem->get_layout();
        auto key_cache_layout = key_cache_mem->get_layout();
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

        // make layouts dynamic
        query_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
        key_layout.set_partial_shape(ov::PartialShape{ -1, p.num_kv_heads * p.k_head_size });
        value_layout.set_partial_shape(ov::PartialShape{ -1, p.num_kv_heads * p.v_head_size });
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
        past_lens_layout.set_partial_shape(ov::PartialShape{ -1 });
        subsequence_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        score_aggregation_window_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotated_block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotation_deltas_layout.set_partial_shape(ov::PartialShape{ -1, -1 });
        rotation_trig_lut_layout.set_partial_shape(ov::PartialShape{ -1, p.k_head_size });
        xattention_threshold_layout.set_partial_shape(ov::PartialShape{ -1 });
        adaptive_rkv_evictable_sizes_layout.set_partial_shape(ov::PartialShape{ -1 });
        adaptive_rkv_diversity_block_set_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        adaptive_rkv_diversity_block_set_indices_begins_layout.set_partial_shape(ov::PartialShape{ -1 });

        if (p.dynamic_paddings) {
            const auto padding_axis = 1;
            const auto pad_before = p.k_head_size;
            const auto pad_after = p.k_head_size * 2;

            query_layout.data_padding._dynamic_dims_mask[padding_axis] = 1;

            auto query_data_layout = query_mem->get_layout();
            auto padded_query_data_layout = query_data_layout;
            padded_query_data_layout.data_padding._lower_size[padding_axis] = pad_before;
            padded_query_data_layout.data_padding._upper_size[padding_axis] = pad_after;

            auto new_query_memory = get_test_engine().allocate_memory(padded_query_data_layout, false);

            mem_lock<ov::float16> query_mem_lock(query_mem, get_test_stream());
            mem_lock<ov::float16> new_query_mem_lock(new_query_memory, get_test_stream());

            auto query_data_shape = query_data_layout.get_shape();
            for (size_t b = 0; b < query_data_shape[0]; b++) {
                for (size_t f = 0; f < query_data_shape[1]; f++) {
                    auto input_offset =
                        query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));
                    auto output_offset =
                        padded_query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));

                    new_query_mem_lock[output_offset] = query_mem_lock[input_offset];
                }
            }
            query_mem = new_query_memory;
        }

        std::vector<input_info> pa_inputs = {
            input_info("query"),
            input_info("key"),
            input_info("value"),
            input_info("key_cache"),
            input_info("value_cache"),
            input_info("past_lens"),
            input_info("subsequence_begins"),
            input_info("block_indices"),
            input_info("block_indices_begins"),
            input_info("scale"),
            input_info("sliding_window"),
            input_info("alibi"),
            input_info("max_context_len"),
            input_info("score_aggregation_window"),
            input_info("rotated_block_indices"),
            input_info("rotation_deltas"),
            input_info("rotation_trig_lut_modified"),
            input_info("xattention_threshold"),
            input_info("xattention_block_size"),
            input_info("xattention_stride"),
            input_info("sinks"),
            input_info("adaptive_rkv_start_size"),
            input_info("adaptive_rkv_evictable_sizes"),
            input_info("adaptive_rkv_diversity_block_set_indices"),
            input_info("adaptive_rkv_diversity_block_set_indices_begins"),
        };

        auto pa_prim = paged_attention("paged_attention", pa_inputs);

        pa_prim.k_head_size = p.k_head_size;
        pa_prim.v_head_size = p.v_head_size;
        pa_prim.kv_heads_num = p.num_kv_heads;
        pa_prim.heads_num = p.num_heads;
        pa_prim.scale_val = pam.get_default_scale();
        pa_prim.has_alibi = false;

        int num_outputs = 1;
        if (p.scores_mode != ScoresMode::DISABLED) num_outputs++;
        if (p.has_adaptive_rkv) num_outputs++;
        pa_prim.num_outputs = num_outputs;
        pa_prim.has_rotated_blocks = p.rotation_config.apply_rotation;
        pa_prim.has_score_aggregation = p.scores_mode == ScoresMode::SNAPKV;
        pa_prim.has_adaptive_rkv = p.has_adaptive_rkv;
        pa_prim.sliding_window = p.sliding_window_size;
        pa_prim.is_key_by_channel = (p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);
        if (p.has_xattention) {
            pa_prim.has_xattention = true;
        }

        topology topology;

        topology.add(
            input_layout("query", query_layout),
            input_layout("key", key_layout),
            input_layout("value", value_layout),
            input_layout("key_cache", key_cache_layout),
            input_layout("value_cache", value_cache_layout),
            input_layout("past_lens", past_lens_layout),
            input_layout("subsequence_begins", subsequence_begins_layout),
            input_layout("block_indices", block_indices_layout),
            input_layout("block_indices_begins", block_indices_begins_layout),
            input_layout("scale", scale_layout),
            input_layout("sliding_window", sliding_window_layout),
            input_layout("alibi", alibi_layout),
            input_layout("max_context_len", max_context_len_layout),
            input_layout("score_aggregation_window", score_aggregation_window_layout),
            pa_prim,
            reorder("output_data", input_info("paged_attention", 0), format::bfyx, data_types::f16)
        );

        int output_idx = 1;
        if (p.scores_mode != ScoresMode::DISABLED) {
            topology.add(reorder("output_scores", input_info("paged_attention", output_idx), format::bfyx, data_types::f16));
            output_idx++;
        }
        if (p.has_adaptive_rkv) {
            topology.add(reorder("output_diversity", input_info("paged_attention", output_idx), format::bfyx, data_types::f16));
        }

        {
            topology.add(input_layout("rotated_block_indices", rotated_block_indices_layout));
            topology.add(input_layout("rotation_deltas", rotation_deltas_layout));
            topology.add(input_layout("rotation_trig_lut", rotation_trig_lut_layout));

            // add dummy activation operation to simulate an empty PA `rotation_trig_lut` buffer for shapes like [0, k_head_size]
            topology.add(activation("rotation_trig_lut_modified", input_info("rotation_trig_lut"), activation_func::none));

            topology.add(input_layout("xattention_threshold", xattention_threshold_layout));
            topology.add(input_layout("xattention_block_size", xattention_block_size_layout));
            topology.add(input_layout("xattention_stride", xattention_stride_layout));
            topology.add(input_layout("sinks", sinks_layout));

            topology.add(input_layout("adaptive_rkv_start_size", adaptive_rkv_start_size_layout));
            topology.add(input_layout("adaptive_rkv_evictable_sizes", adaptive_rkv_evictable_sizes_layout));
            topology.add(input_layout("adaptive_rkv_diversity_block_set_indices", adaptive_rkv_diversity_block_set_indices_layout));
            topology.add(input_layout("adaptive_rkv_diversity_block_set_indices_begins", adaptive_rkv_diversity_block_set_indices_begins_layout));
        }

        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        // FlashAttn v1 or v2?
        config.set_property(ov::intel_gpu::could_use_flashattn_v2(p.disable_flashattn_v2));
        config.set_property(ov::internal::key_cache_quant_mode(p.key_cache_quant_mode));
        network::ptr network = get_network(get_test_engine(), topology, config, get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);
        network->set_input_data("key_cache", key_cache_mem);
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

        auto outputs = network->execute();

        cldnn::memory::ptr output_data_mem = nullptr;
        cldnn::memory::ptr output_scores_mem = nullptr;
        cldnn::memory::ptr output_diversity_mem = nullptr;

        output_data_mem = outputs.at("output_data").get_memory();
        if (p.scores_mode != ScoresMode::DISABLED) {
            output_scores_mem = outputs.at("output_scores").get_memory();
        }
        if (p.has_adaptive_rkv) {
            output_diversity_mem = outputs.at("output_diversity").get_memory();
        }
        auto ref_data = PagedAttentionReference(pam).get_reference(p.has_xattention, p.threshold, key_cache_mem);
        if (p.has_xattention) {
            compare_xattention(output_data_mem, output_scores_mem, ref_data);
        } else {
            compare(output_data_mem, output_scores_mem, output_diversity_mem, ref_data);
        }
    }

    void compare(memory::ptr data_output_mem, memory::ptr scores_output_mem, memory::ptr diversity_output_mem, std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> ref_data) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), std::get<0>(ref_data).size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output_mem, get_test_stream());
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<0>(ref_data)[i], tolerance) << " at index=" << i;
            }
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), std::get<1>(ref_data).size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output_mem, get_test_stream());
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<1>(ref_data)[i], tolerance) << " at index=" << i;
            }
        }

        if (diversity_output_mem) {
            ASSERT_EQ(diversity_output_mem->count(), std::get<2>(ref_data).size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(diversity_output_mem, get_test_stream());
            // Relaxed tolerance due to float32 (GPU) vs float16 (reference) accumulator difference
            float diversity_tolerance = tolerance * 10.0f;
            for (size_t i = 0; i < diversity_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], std::get<2>(ref_data)[i], diversity_tolerance) << " at index=" << i;
            }
        }
    }

    void compare_xattention(memory::ptr data_output_mem, memory::ptr scores_output_mem, std::tuple<std::vector<ov::float16>, std::vector<ov::float16>, std::vector<ov::float16>> ref_data) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), std::get<0>(ref_data).size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output_mem, get_test_stream());
            int mismatch_count = 0;
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                if (std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(std::get<0>(ref_data)[i])) > tolerance) {
                    mismatch_count++;
                }
            }
            EXPECT_LE(mismatch_count, int(data_output_mem->count() * 0.04));
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), std::get<1>(ref_data).size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output_mem, get_test_stream());
            int mismatch_count = 0;
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                if (std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(std::get<1>(ref_data)[i])) > tolerance) {
                    mismatch_count++;
                }
            }
            EXPECT_LE(mismatch_count, int(scores_output_mem->count() * 0.04));
        }
    }

    static bool check_cm_available() {
        auto& engine = get_test_engine();
        ExecutionConfig config = get_test_default_config(engine);
        return cldnn::check_cm_jit_support(engine, config) &&
               (engine.get_device_info().arch == gpu_arch::xe2 || engine.get_device_info().arch == gpu_arch::xe3);
    }
};

struct paged_attention_test_params {
    std::vector<SubsequenceDescriptor> subsequences;
    int num_heads;
    int num_kv_heads;
    int k_head_size;
    int v_head_size;
    int block_size;
    std::vector<float> threshold;
    int sliding_window_size;
    bool has_xattention;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    bool dynamic_paddings;
    ScoresMode scores_mode;
    CacheRotationDescriptor rotation_config;
    bool disable_flashattn_v2;
    bool has_adaptive_rkv = false;
    int start_size = 0;                // Common start_size for all sequences
    std::vector<int> evictable_sizes;  // Per-sequence evictable sizes
};

class paged_attention_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(paged_attention_test, basic) {
    auto p = GetParam();

    execute(p);
}

class xattention_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(xattention_test, basic) {
    if (!check_cm_available())
        GTEST_SKIP();
    auto p = GetParam();

    execute(p);
}

class adaptive_rkv_diversity_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(adaptive_rkv_diversity_test, basic) {
    auto p = GetParam();

    execute(p);
}

const auto ENABLE_CACHE_COMPRESSION = true;
const auto DISABLE_CACHE_COMPRESSION = false;
const auto DISABLE_SCORES = ScoresMode::DISABLED;
const auto ENABLE_SCORES = ScoresMode::LAST_TOKEN;
const auto ENABLE_SCORES_SNAPKV = ScoresMode::SNAPKV;
const auto PER_BLOCK_ROTATION = CacheRotationDescriptor{ true, true };
const auto PER_TOKEN_ROTATION = CacheRotationDescriptor{ true, false };
const auto DISABLE_ROTATION = CacheRotationDescriptor{ false, false };
const auto STATIC_INPUT_PAD = false;
const auto DYNAMIC_INPUT_PAD = true;
const auto ENABLE_FA_V2 = false;
const auto DISABLE_FA_V2 = true;
const auto ENABLE_DIVERSITY = true;
const auto DISABLE_DIVERSITY = false;

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention, paged_attention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    /* with scores output, use SnapKV */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores output */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 16, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 128, 96, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 48, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 96, 128, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 32, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 96, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 48, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 16, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_block rotation */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 48, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_token rotation */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 128, 192, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings, KV-cache compression */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64,  16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64,  16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32,  16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 32,  16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token

    /* with scores, per_block rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    /* with scores, per_token rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* With sliding windows */
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 6, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 2, 64, 64, 16, {100.0}, 6, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 2, 64, 32, 16, {100.0}, 20, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 2, 64, 32, 16, {100.0}, 20, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 2, 64, 64, 16, {100.0}, 8, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 2, 48, 64, 16, {100.0}, 128, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 2, 64, 64, 16, {100.0}, 4, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{5, 10}}, 2, 2, 64, 64, 16, {100.0}, 2, false, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{248, 16}}, 2, 2, 128, 128, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{34, 0}}, 2, 2, 32, 32, 16, {100.0}, 2, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1, 1008}}, 32, 32, 128, 128, 16, {100.0}, 6, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{6, 20}}, 2, 2, 128, 128, 16, {100.0}, 8, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1, 288}}, 64, 8, 64, 64, 16, {100.0}, 128, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching, GQA
    paged_attention_test_params{ {{84, 2}}, 32, 32, 128, 128, 16, {100.0}, 16, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 32, 32, 32, 32, 16, {100.0}, 32, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 16, 16, 64, 64, 16, {100.0}, 64, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 492}}, 8, 8, 128, 128, 16, {100.0}, 128, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{2, 30}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{232, 24}}, 2, 2, 512, 512, 16, {100.0}, 32, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 592}}, 32, 32, 128, 128, 16, {100.0}, 64, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 692}}, 32, 32, 128, 128, 16, {100.0}, 128, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1008, 792}}, 32, 32, 128, 128, 16, {100.0}, 256, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: prefix caching
    paged_attention_test_params{ {{1, 34}, {2, 20}, {10, 34}}, 2, 2, 64, 64, 16, {100.0}, 10, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* Per-Block rotation cases with  Per-Channel quantization*/
    paged_attention_test_params{ {{16, 32}}, 2, 2, 32, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{8, 34}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{256, 56}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{48, 1024}}, 8, 8, 128, 128, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{2, 34}, {2, 515}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{4, 34}, {4, 515}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{6, 34}, {25, 86}, {10, 64}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate
    paged_attention_test_params{ {{8, 34}, {25, 86}, {10, 64}}, 2, 2, 64, 32, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token, per block rotate

    /* Per-Token rotation cases with  Per-Channel quantization*/
    paged_attention_test_params{ {{8, 38}}, 16, 16, 256, 256, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{34, 34}}, 2, 2, 32, 32, 16, {100.0}, 2, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{2, 1008}}, 32, 32, 128, 128, 16, {100.0}, 6, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{6, 40}}, 2, 2, 128, 128, 16, {100.0}, 8, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{254, 60}}, 32, 8, 128, 128, 16, {100.0}, 10, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{84, 256}}, 32, 32, 128, 128, 16, {100.0}, 16, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{40, 96}, {30, 50}}, 2, 2, 64, 64, 16, {100.0}, 8, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token, per token rotate
    paged_attention_test_params{ {{128, 130}, {256, 60}}, 2, 2, 48, 64, 16, {100.0}, 128, false, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, ENABLE_FA_V2 }, // 2nd token, per token rotate
}));

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention, xattention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    paged_attention_test_params{ {{32, 0}},   2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{2048, 0}}, 2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token

    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 1023}}, 2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 127}},  2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 128}},  2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 129}},  2, 2, 64, 64, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 32}},   28, 28, 128, 128, 256, {0.9}, 0, true, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token

    paged_attention_test_params{ {{32, 0}},   2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{2048, 0}}, 2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{32, 0}},   4, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 4, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{2048, 0}}, 4, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token

    paged_attention_test_params{ {{1, 31}},   2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 32}},   2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd toke
    paged_attention_test_params{ {{1, 1023}},   2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 1024}}, 2, 2, 64, 64, 256, {0.9}, 0, true, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
}));

INSTANTIATE_TEST_SUITE_P(smoke_adaptive_rkv, adaptive_rkv_diversity_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{
    // Small evictable_size tests (uniform across sequences)
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },

    // Edge cases - empty evictable_size (kernel skip test)
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 0, {0} },    // Empty - skip kernel
    paged_attention_test_params{ {{32, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {16} },  // Minimal valid size (block-aligned)

    // Multi-sequence tests - uniform evictable_sizes
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },            // 2 sequences: same size
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },  // 3 sequences: same size

    // Multi-sequence tests - varying evictable_sizes per sequence
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },                              // 2 sequences: different evictable sizes
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },                // 3 sequences: different sizes
    paged_attention_test_params{ {{128, 0}, {128, 0}, {128, 0}, {128, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {16, 32, 48, 32} },  // 4 sequences: varied sizes

    // With KV cache compression - BY_TOKEN
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },

    // With KV cache compression - BY_CHANNEL
    paged_attention_test_params{ {{64, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32} },
    paged_attention_test_params{ {{128, 0}}, 4, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },
    paged_attention_test_params{ {{128, 0}, {128, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48} },
    paged_attention_test_params{ {{128, 0}, {192, 0}, {160, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, ENABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 16, {32, 48, 32} },

    // Large evictable_size tests
    paged_attention_test_params{ {{192, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 48, {96} },
    paged_attention_test_params{ {{256, 0}}, 2, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 64, {128} },
    paged_attention_test_params{ {{512, 0}}, 4, 2, 128, 128, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 128, {256} },

    // GQA configurations
    paged_attention_test_params{ {{128, 0}}, 8, 2, 64, 64, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },     // heads_num=8, start+evict=96
    paged_attention_test_params{ {{160, 0}}, 16, 4, 128, 128, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {48} },  // heads_num=16, start+evict=80

    // Different head sizes
    paged_attention_test_params{ {{128, 0}}, 2, 2, 128, 128, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {64} },  // head_size=128, start+evict=96
    paged_attention_test_params{ {{160, 0}}, 4, 4, 256, 256, 16, {100.0}, 0, false, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2, ENABLE_DIVERSITY, 32, {48} },  // head_size=256, start+evict=80
}));
