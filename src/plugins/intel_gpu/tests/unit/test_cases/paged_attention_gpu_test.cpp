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
                int blocks_num = ceil_div(past_len, block_size);
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
                int blocks_num = ceil_div(past_len, block_size);
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

    std::pair<std::vector<ov::float16>, std::vector<ov::float16>> get_reference() {
        std::vector<ov::float16> ref_data_output;
        std::vector<ov::float16> ref_scores_output;

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
                                     pam.num_heads,
                                     pam.k_head_size,
                                     pam.block_size,
                                     pam.rotation_config.per_block);
                    }
                }
            }

            auto window_size = pam.has_score_aggregation ? pam.score_aggregation[i] : 1;

            auto subsequence_ref_results = run_reference(pam.query_data[i],
                                                         key_data,
                                                         pam.value_data[i],
                                                         subsequence_desc.num_tokens,
                                                         kv_seq_len,
                                                         pam.num_heads,
                                                         pam.k_head_size,
                                                         pam.v_head_size,
                                                         window_size,
                                                         pam.sliding_window_size,
                                                         pam.get_default_scale());

            // concatenate all subsequences into one vector
            ref_data_output.insert(ref_data_output.end(),
                                   subsequence_ref_results.first.begin(),
                                   subsequence_ref_results.first.end());
            ref_scores_output.insert(ref_scores_output.end(),
                                     subsequence_ref_results.second.begin(),
                                     subsequence_ref_results.second.end());
        }

        return { ref_data_output, ref_scores_output };
    }

private:
    std::pair<std::vector<ov::float16>, std::vector<ov::float16>>
        run_reference(const std::vector<ov::float16>& query_data,
                      const std::vector<ov::float16>& key_data,
                      const std::vector<ov::float16>& value_data,
                      int num_queries,
                      int num_keys,
                      int num_heads,
                      int k_head_size,
                      int v_head_size,
                      int window_size,
                      int sliding_window_size,
                      float scale) {
        auto query_shape = ov::PartialShape{1, num_queries, num_heads, k_head_size};
        auto key_shape = ov::PartialShape{1, num_keys, num_heads, k_head_size};
        auto value_shape = ov::PartialShape{1, num_keys, num_heads, v_head_size};

        auto query_layout = layout{query_shape, data_types::f16, format::bfyx};
        auto key_layout = layout{key_shape, data_types::f16, format::bfyx};
        auto value_layout = layout{value_shape, data_types::f16, format::bfyx};

        OPENVINO_ASSERT(query_layout.count() == query_data.size());
        OPENVINO_ASSERT(key_layout.count() == key_data.size());
        OPENVINO_ASSERT(value_layout.count() == value_data.size());

        auto query_mem = test_engine.allocate_memory(query_layout);
        auto key_mem = test_engine.allocate_memory(key_layout);
        auto value_mem = test_engine.allocate_memory(value_layout);
        auto mask_mem = get_mask_mem(num_queries, num_keys, num_heads, sliding_window_size);

        set_values(query_mem, query_data);
        set_values(key_mem, key_data);
        set_values(value_mem, value_data);

        topology topology;
        topology.add(input_layout("query", query_layout),
                     input_layout("key", key_layout),
                     input_layout("value", value_layout),
                     data("mask", mask_mem),
                     permute("query_transposed", input_info("query"), {0, 2, 1, 3}),
                     permute("key_transposed", input_info("key"), {0, 2, 1, 3}),
                     permute("value_transposed", input_info("value"), {0, 2, 1, 3}),
                     gemm("qk_gemm", { input_info("query_transposed"), input_info("key_transposed") }, data_types::f16, false, true, scale),
                     eltwise("eltwise", { input_info("qk_gemm"), input_info("mask") }, eltwise_mode::sum),
                     softmax("softmax", input_info("eltwise"), -1),
                     gemm("qkv_gemm", { input_info("softmax"), input_info("value_transposed") }, data_types::f16, false, false),
                     permute("qkv_gemm_transposed", input_info("qkv_gemm"), {0, 2, 1, 3}),
                     reorder("output_data", input_info("qkv_gemm_transposed"), format::bfyx, data_types::f16),
                     reorder("scores_data", input_info("softmax"), format::bfyx, data_types::f16)
        );

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

    memory::ptr get_mask_mem(int num_queries, int num_keys, int num_heads, int sliding_window_size) {
        /*
        * Two kinds of masks:
        *
        * Case 1 (N == K):
        * num_queries = N
        * num_keys = K = N
        * k_head_size = H
        * Q  [N, H] * K[H, N]
        * QK [N, N]
        *       0    1        N
        * 0  [  0, MIN, .., MIN ]
        * 1  [  0,   0, .., MIN ]
        *    [ ..,  .., .., MIN ]
        * N  [  0,   0, ..,   0 ]
        *
        * Case 2 (N != K):
        * num_queries = N
        * num_keys = K
        * k_head_size = H
        * past_len = P = K - N + 1
        * Q  [N, H] * K[H, K]
        * QK [N, K]
        *      0    1    2    P   ..    K
        * 0 [  0,   0,   0, MIN, MIN, MIN ]
        * 1 [  0,   0,   0,   0, MIN, MIN ]
        *   [  .., ..,  ..,  ..,  .., MIN ]
        * N [  0,   0,   0,   0,  ..,   0 ]
        *
        * Shapes:
        * Q   [1, num_heads, num_queries, k_head_size]
        * K   [1, num_heads, k_head_size, num_keys]
        * Q*K [1, num_heads, num_queries, num_keys]
        */

        auto mask_shape = ov::PartialShape{ 1, 1, num_queries, num_keys };
        auto mask_layout = layout{mask_shape, data_types::f16, format::bfyx};
        auto mask_mem = test_engine.allocate_memory(mask_layout);

        mem_lock<ov::float16> mem_ptr(mask_mem, test_stream);

        if (sliding_window_size == 0) {
            int past_len = num_keys - num_queries + 1;
            for (int i = 0; i < num_queries; i++) {
                for (int j = 0; j < num_keys; j++) {
                    mem_ptr[i * num_keys + j] = j >= past_len + i ? std::numeric_limits<ov::float16>::lowest()
                                                                    : ov::float16(0.f);
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
                                  p.k_head_size,
                                  p.v_head_size,
                                  p.block_size,
                                  p.sliding_window_size,
                                  p.kv_cache_compression,
                                  p.key_cache_quant_mode,
                                  p.scores_mode == ScoresMode::SNAPKV,
                                  p.rotation_config);

        if (p.kv_cache_compression)
            tolerance = 25e-3;

        auto query_mem = pam.get_query_memory();
        auto key_mem = pam.get_key_memory();
        auto value_mem = pam.get_value_memory();

        auto key_cache_mem = pam.get_key_cache_memory();
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

        // make layouts dynamic
        query_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
        key_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
        value_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.v_head_size });
        key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.k_head_size, p.block_size });
        value_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.block_size, p.v_head_size });
        past_lens_layout.set_partial_shape(ov::PartialShape{ -1 });
        subsequence_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        score_aggregation_window_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotated_block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotation_deltas_layout.set_partial_shape(ov::PartialShape{ -1, -1 });
        rotation_trig_lut_layout.set_partial_shape(ov::PartialShape{ -1, p.k_head_size });
        xattention_threshold_layout.set_partial_shape(ov::PartialShape{ -1 });

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
        };

        auto pa_prim = paged_attention("paged_attention", pa_inputs);

        pa_prim.k_head_size = p.k_head_size;
        pa_prim.v_head_size = p.v_head_size;
        pa_prim.kv_heads_num = p.num_heads;
        pa_prim.heads_num = p.num_heads;
        pa_prim.scale_val = pam.get_default_scale();
        pa_prim.has_alibi = false;
        pa_prim.num_outputs = p.scores_mode == ScoresMode::DISABLED ? 1 : 2;
        pa_prim.has_rotated_blocks = p.rotation_config.apply_rotation;
        pa_prim.has_score_aggregation = p.scores_mode == ScoresMode::SNAPKV;
        pa_prim.sliding_window = p.sliding_window_size;
        pa_prim.is_key_by_channel = (p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);

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

        if (p.scores_mode != ScoresMode::DISABLED) {
            topology.add(reorder("output_scores", input_info("paged_attention", 1), format::bfyx, data_types::f16));
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

        auto outputs = network->execute();

        cldnn::memory::ptr output_data_mem = nullptr;
        cldnn::memory::ptr output_scores_mem = nullptr;

        output_data_mem = outputs.at("output_data").get_memory();
        if (p.scores_mode != ScoresMode::DISABLED) {
            output_scores_mem = outputs.at("output_scores").get_memory();
        }
        auto ref_data = PagedAttentionReference(pam).get_reference();
        compare(output_data_mem, output_scores_mem, ref_data);
    }

    void compare(memory::ptr data_output_mem, memory::ptr scores_output_mem, std::pair<std::vector<ov::float16>, std::vector<ov::float16>> ref_data) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), ref_data.first.size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output_mem, get_test_stream());
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], ref_data.first[i], tolerance) << " at index=" << i;
            }
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), ref_data.second.size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output_mem, get_test_stream());
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                ASSERT_NEAR(mem_ptr[i], ref_data.second[i], tolerance) << " at index=" << i;
            }
        }
    }
};

struct paged_attention_test_params {
    std::vector<SubsequenceDescriptor> subsequences;
    int num_heads;
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
};

class paged_attention_test : public PagedAttentionTest<paged_attention_test_params> {};
TEST_P(paged_attention_test, basic) {
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

INSTANTIATE_TEST_SUITE_P(smoke_paged_attention, paged_attention_test, ::testing::ValuesIn(std::vector<paged_attention_test_params>{

#if ENABLE_PA_CM_PATH
    /* without scores output, dynamic input query paddings */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token

    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
#else
    /* with scores output, use SnapKV */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES_SNAPKV, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores output */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 16, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 128, 96, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 96, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 32, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 48, 96, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 16, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_block rotation */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN,STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{36, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 48, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 48, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* with scores, per_token rotation */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 128, 192, 16, 0, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* without scores output, dynamic input query paddings, KV-cache compression */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 32,  16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token long
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{10, 0}, {81, 0}, {129, 0}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token

    /* with scores, per_block rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, DISABLE_FA_V2 }, // 2nd token
    /* with scores, per_token rotation, KV-cache compression */
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {1, 515}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 64, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    paged_attention_test_params{ {{1, 34}, {25, 0}, {10, 34}}, 2, 64, 32, 16, 0, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
    /* With sliding windows */
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}}, 2, 64, 64, 16, 6, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 64, 32, 16, 20, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{512, 0}}, 2, 64, 32, 16, 20, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, DYNAMIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{10, 0}, {30, 0}}, 2, 64, 64, 16, 8, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // 2nd token + 2nd token
    paged_attention_test_params{ {{128, 0}, {256, 0}}, 2, 48, 64, 16, 128, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_BLOCK_ROTATION, ENABLE_FA_V2 }, // 1st token + 1st token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 64, 16, 4, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 10}}, 2, 64, 64, 16, 4, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{5, 10}}, 2, 64, 64, 16, 2, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{5, 10}}, 2, 64, 64, 16, 2, DISABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_CHANNEL, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 34}, {2, 20}, {10, 34}}, 2, 64, 64, 16, 10, ENABLE_CACHE_COMPRESSION,ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, ENABLE_SCORES, PER_TOKEN_ROTATION, DISABLE_FA_V2 }, // mixed: 2nd token + 1st token + part of 1st token
#endif
}));
