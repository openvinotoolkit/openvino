// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "paged_attention_gpu_test.hpp"

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
#if ENABLE_PA_CM_PATH
        key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.block_size, p.k_head_size });
#else
        key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.k_head_size, p.block_size });
#endif
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
            // for (size_t i = 0; i < data_output_mem->count(); i++) {
            //     std::cout << i << ": result = " << mem_ptr[i] << ", reference = " << ref_data.first[i] << std::endl;
            // }
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
     /* without scores output, static input query paddings, single sequence, disable KV cache compression, k_head_size==v_head_size, token_size>=32, disable_mix_mode */
    paged_attention_test_params{ {{32, 0}},   2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    paged_attention_test_params{ {{1024, 0}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 1st token long

    paged_attention_test_params{ {{1, 31}},   2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 32}},   2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 1023}}, 2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 127}},  2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 129}},  2, 64, 64, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    paged_attention_test_params{ {{1, 32}},  28, 128, 128, 16, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
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
