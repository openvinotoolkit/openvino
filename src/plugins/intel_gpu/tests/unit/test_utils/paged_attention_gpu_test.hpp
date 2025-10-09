// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/softmax.hpp>
#include <openvino/reference/xattention.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

/*
* PagedAttention inputs:
* [0]: query
* shape: [batch_size_in_tokens, num_heads * head_size], type: f16
* [1]: key
* shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
* [2]: value 
* shape: [batch_size_in_tokens, num_kv_heads * head_size], type: f16
* [3]: key_cache
* shape: [num_blocks, num_kv_heads, head_size, block_size], type: f16
* [4]: value_cache
* shape: [num_blocks, num_kv_heads, block_size, head_size], type: f16
* [5]: past_lens
* shape: [batch_size_in_sequences], type: i32
* [6]: subsequence_begins
* shape: [batch_size_in_sequences + 1], type: i32
* [7]: block_indices
* Shape: [num_blocks], type: i32
* [8]: block_indices_begins
* Shape: [batch_size_in_sequences + 1], type: i32
* [9]: scale, optional
* [10]: sliding_window, optional
* [11]: alibi_slopes, optional
* [12]: max_context_len
* shape: [], type: i32
* [13]: score_aggregation_window​, optional​, shape: [batch_size_in_sequences]
* [14]: rotated_block_indices​, optional​
* shape: [num_rotated_blocks]​, type: i32
* [15]: rotation_deltas​, optional​
* shape: [num_rotated_blocks, BLOCK_SIZE]​ || [num_rotated_blocks, 1]​, type: i32
* [16]: rotation_trig_lut​, optional​
* shape: [max_num_batched_tokens / BLOCK_SIZE, head_size]​ || [max_num_batched_tokens, head_size], type: f16
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
    int k_head_size;
    int v_head_size;
    int block_size;
    int sliding_window_size;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    bool has_score_aggregation;
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

    cldnn::engine& test_engine;
    cldnn::stream& test_stream;
    tests::random_generator& rg;

    PagedAttentionManager(tests::random_generator& rg,
                          cldnn::engine& engine,
                          cldnn::stream& stream,
                          const std::vector<SubsequenceDescriptor>& subsequence_descs,
                          int num_heads,
                          int k_head_size,
                          int v_head_size,
                          int block_size,
                          int sliding_window_size,
                          bool kv_cache_compression,
                          ov::internal::CacheQuantMode key_cache_quant_mode,
                          bool has_score_aggregation,
                          CacheRotationDescriptor rotation_config)
        : num_heads(num_heads)
        , k_head_size(k_head_size)
        , v_head_size(v_head_size)
        , block_size(block_size)
        , sliding_window_size(sliding_window_size)
        , kv_cache_compression(kv_cache_compression)
        , key_cache_quant_mode(key_cache_quant_mode)
        , has_score_aggregation(has_score_aggregation)
        , rotation_config(rotation_config)
        , subsequence_descs(subsequence_descs)
        , test_engine(engine)
        , test_stream(stream)
        , rg(rg) {
        // init subsequence_begins and block_indices_begins
        subsequence_begins.push_back(0);
        block_indices_begins.push_back(0);

        int max_len = 0;
        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            const auto& subsequence_desc = subsequence_descs[i];
            max_len = std::max(max_len, subsequence_desc.num_tokens + subsequence_desc.past_len);

            query_data.push_back(generate_input_data(rg, num_heads, subsequence_desc.num_tokens, k_head_size));
            key_data.push_back(generate_input_data(rg, num_heads, subsequence_desc.num_tokens + subsequence_desc.past_len, k_head_size));
            value_data.push_back(generate_input_data(rg, num_heads, subsequence_desc.num_tokens + subsequence_desc.past_len, v_head_size));

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
        return get_QKV_memory(query_data, k_head_size, false);
    }

    memory::ptr get_key_memory() {
        return get_QKV_memory(key_data, k_head_size, true);
    }

    memory::ptr get_value_memory() {
        return get_QKV_memory(value_data, v_head_size, true);
    }

#if ENABLE_PA_CM_PATH
    memory::ptr get_key_cache_memory() {
        auto key_cache_dt = data_types::f16;
        auto adjusted_head_size = k_head_size;
        if (kv_cache_compression) {
            key_cache_dt = data_types::i8;
            adjusted_head_size += 4;
        }

        auto num_blocks = block_indices.back() + 1;
        auto key_cache_shape = ov::PartialShape{ num_blocks, num_heads, block_size, adjusted_head_size };
        auto key_cache_layout = layout{ key_cache_shape, key_cache_dt, format::bfyx };
        auto memory = test_engine.allocate_memory(key_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            int past_len = subsequence_descs[i].past_len;
            if (past_len != 0) {
                int blocks_num = ceil_div(past_len + 1, block_size);
                int start_block_idx = block_indices[block_indices_begins[i]];
                for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                    int last_token_idx = block_idx == blocks_num - 1 ? past_len % block_size
                                                                     : block_size;
                    for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                            size_t input_token_offset = block_idx * block_size + token_idx;
                            ov::float16* data_ptr = key_data[i].data() +
                                                    input_token_offset * num_heads * v_head_size +
                                                    head_idx * v_head_size;
                            if (kv_cache_compression) {
                                auto [quantized_data, scale, zp] = quantize_data(data_ptr, v_head_size);
                                auto quantized_data_ptr = quantized_data.data();

                                // shape: [num_blocks, num_heads, block_size, adjusted_head_size]
                                size_t output_block_offset = (start_block_idx + block_idx) * num_heads * block_size * adjusted_head_size +
                                                             head_idx * block_size * adjusted_head_size;
                                size_t output_offset = output_block_offset +
                                                       token_idx * v_head_size;
                                set_values(test_stream, memory, quantized_data_ptr, v_head_size, output_offset);

                                size_t comp_offset = (output_block_offset + v_head_size * block_size) / 2;
                                set_values(test_stream, memory, &scale, 1, comp_offset + token_idx);
                                set_values(test_stream, memory, &zp, 1, comp_offset + block_size + token_idx);
                            } else {
                                // shape: [num_blocks, num_heads, block_size, v_head_size]
                                size_t output_offset = (start_block_idx + block_idx) * num_heads * block_size * v_head_size +
                                                       head_idx * block_size * v_head_size +
                                                       token_idx * v_head_size;

                                set_values(test_stream, memory, data_ptr, v_head_size, output_offset);
                            }
                        }
                    }
                }
            }
        }

        return memory;
    }

#else
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
        auto key_cache_shape = ov::PartialShape{ num_blocks, num_heads, adjusted_head_size, adjusted_block_size };
        auto key_cache_layout = layout{ key_cache_shape, key_cache_dt, format::bfyx };
        auto memory = test_engine.allocate_memory(key_cache_layout);
        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            int past_len = subsequence_descs[i].past_len;
            if (past_len != 0) {
                int blocks_num = ceil_div(past_len + 1, block_size);
                int start_block_idx = block_indices[block_indices_begins[i]];
                for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                    int last_token_idx = block_idx == blocks_num - 1 ? past_len % block_size
                                                                     : block_size;
                    // quantize by channel
                    if (kv_cache_compression && key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL) {
                        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                            for (int k_head_size_idx = 0; k_head_size_idx < k_head_size; k_head_size_idx++) {
                                std::vector<ov::float16> token_block(block_size);
                                for (int token_idx = 0; token_idx < last_token_idx; ++token_idx) {
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    token_block[token_idx] = *(key_data[i].data() + input_token_offset * num_heads * k_head_size + head_idx * k_head_size + k_head_size_idx);
                                }
                                auto [quantized_data, scale, zp] = quantize_data(token_block.data(), last_token_idx, true);
                                size_t output_block_offset = (start_block_idx + block_idx) * num_heads * adjusted_head_size * adjusted_block_size +
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
                        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                            if (kv_cache_compression) {
                                if (key_cache_quant_mode == ov::internal::CacheQuantMode::BY_TOKEN) {
                                    // quantize by token
                                    size_t input_token_offset = block_idx * block_size + token_idx;
                                    ov::float16* data_ptr = key_data[i].data() +
                                                            input_token_offset * num_heads * k_head_size +
                                                            head_idx * k_head_size;
                                    // shape: [num_blocks, num_heads, adjusted_head_size, block_size]
                                    size_t output_block_offset = (start_block_idx + block_idx) * num_heads * adjusted_head_size * block_size +
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
                                                            input_token_offset * num_heads * k_head_size +
                                                            head_idx * k_head_size + k_head_size_idx;

                                    // shape: [num_blocks, num_heads, k_head_size, block_size]
                                    size_t output_offset = (start_block_idx + block_idx) * num_heads * k_head_size * block_size +
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
#endif

    memory::ptr get_value_cache_memory() {
        auto value_cache_dt = data_types::f16;
        auto adjusted_head_size = v_head_size;
        if (kv_cache_compression) {
            value_cache_dt = data_types::i8;
            adjusted_head_size += 4;
        }

        auto num_blocks = block_indices.back() + 1;
        auto value_cache_shape = ov::PartialShape{ num_blocks, num_heads, block_size, adjusted_head_size };
        auto value_cache_layout = layout{ value_cache_shape, value_cache_dt, format::bfyx };
        auto memory = test_engine.allocate_memory(value_cache_layout);

        for (int i = 0; i < static_cast<int>(subsequence_descs.size()); i++) {
            int past_len = subsequence_descs[i].past_len;
            if (past_len != 0) {
                int blocks_num = ceil_div(past_len + 1, block_size);
                int start_block_idx = block_indices[block_indices_begins[i]];
                for (int block_idx = 0; block_idx < blocks_num; block_idx++) {
                    int last_token_idx = block_idx == blocks_num - 1 ? past_len % block_size
                                                                     : block_size;
                    for (int token_idx = 0; token_idx < last_token_idx; token_idx++) {
                        for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                            size_t input_token_offset = block_idx * block_size + token_idx;
                            ov::float16* data_ptr = value_data[i].data() +
                                                    input_token_offset * num_heads * v_head_size +
                                                    head_idx * v_head_size;
                            if (kv_cache_compression) {
                                auto [quantized_data, scale, zp] = quantize_data(data_ptr, v_head_size);
                                auto quantized_data_ptr = quantized_data.data();

                                // shape: [num_blocks, num_heads, block_size, adjusted_head_size]
                                size_t output_block_offset = (start_block_idx + block_idx) * num_heads * block_size * adjusted_head_size +
                                                             head_idx * block_size * adjusted_head_size;
                                size_t output_offset = output_block_offset +
                                                       token_idx * v_head_size;
                                set_values(test_stream, memory, quantized_data_ptr, v_head_size, output_offset);

                                size_t comp_offset = (output_block_offset + v_head_size * block_size) / 2;
                                set_values(test_stream, memory, &scale, 1, comp_offset + token_idx);
                                set_values(test_stream, memory, &zp, 1, comp_offset + block_size + token_idx);
                            } else {
                                // shape: [num_blocks, num_heads, block_size, v_head_size]
                                size_t output_offset = (start_block_idx + block_idx) * num_heads * block_size * v_head_size +
                                                       head_idx * block_size * v_head_size +
                                                       token_idx * v_head_size;

                                set_values(test_stream, memory, data_ptr, v_head_size, output_offset);
                            }
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

    memory::ptr get_QKV_memory(std::vector<std::vector<ov::float16>>& input_data, int k_head_size, bool skip_past_len) {
        int total_tokens = 0;
        for (const auto& subsequence_desc : subsequence_descs)
            total_tokens += subsequence_desc.num_tokens;

        auto query_shape = ov::PartialShape{ total_tokens, num_heads * k_head_size };
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
                                            input_token_offset * num_heads * k_head_size +
                                            head_idx * k_head_size;

                    size_t output_token_offset = subsequence_begins[subsequence_idx] + token_idx;
                    size_t output_offset = output_token_offset * num_heads * k_head_size +
                                           head_idx * k_head_size;

                    set_values(test_stream, memory, data_ptr, k_head_size, output_offset);
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

        // test code
        // auto data = rg.generate_random_1d_fixed<ov::float16>(total_elements_num, 0, 1, 10000);

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

    static std::tuple<std::vector<int8_t>, ov::float16, ov::float16> quantize_data(ov::float16* data, size_t size, bool expand_range = false) {
        float min_value = std::numeric_limits<float>::max();
        float max_value = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < size; i++) {
            min_value = std::min((float)(data[i]), min_value);
            max_value = std::max((float)(data[i]), max_value);
        }

        float diff_value = 0.001;
        if (max_value != min_value)
            diff_value = max_value - min_value;
        if (expand_range && std::abs(diff_value) <= std::abs(max_value) * 0.1f) {
            // compensate too small range
            diff_value = (max_value - min_value) + std::max(1.0f, max_value * 0.1f);
        }
        float scale = (std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::lowest()) / diff_value;
        float zp = ((float)-min_value * scale) + std::numeric_limits<int8_t>::lowest();

        std::vector<int8_t> quantized_data;
        quantized_data.resize(size);

        auto convert_char_rte = [](float val) {
            float rounded = std::nearbyint(val);

            if (rounded > 127.0f) {
                return static_cast<int8_t>(127);
            } else if (rounded < -128.0f) {
                return static_cast<int8_t>(-128);
            } else {
                return static_cast<int8_t>(rounded);
            }
        };

        for (size_t i = 0; i < size; i++) {
            quantized_data[i] = convert_char_rte(data[i] * scale + zp);
        }

        scale = 1.0f / scale;

        return std::make_tuple(quantized_data, scale, zp);
    }
};
