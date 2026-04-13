// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_gated_delta_net.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

void normalize_and_scale(const float* src, size_t n, float scale, std::vector<float>& dst) {
    dst.resize(n);
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        dst[i] = src[i];
        sum += src[i] * src[i];
    }
    const float inv = 1.0f / std::sqrt(sum + 1e-6f);
    for (size_t i = 0; i < n; i++) {
        dst[i] *= inv * scale;
    }
}

void run_reference(const std::vector<float>& query,
                   const std::vector<float>& key,
                   const std::vector<float>& value,
                   const std::vector<float>& gate,
                   const std::vector<float>& beta,
                   std::vector<float>& recurrent_state_table,
                   const std::vector<int32_t>& subsequence_begins,
                   const std::vector<int32_t>& block_indices,
                   const std::vector<int32_t>& block_indices_begins,
                   [[maybe_unused]] const std::vector<int32_t>& past_lens,
                   const std::vector<int32_t>& cache_interval,
                   int32_t qk_heads,
                   int32_t v_heads,
                   int32_t qk_head_size,
                   int32_t v_head_size,
                   std::vector<float>& output) {
    const int32_t tokens = static_cast<int32_t>(query.size()) / (qk_heads * qk_head_size);
    const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
    const int32_t group_size = v_heads / qk_heads;
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(qk_head_size));

    output.resize(static_cast<size_t>(tokens) * v_heads * v_head_size);

    // state layout: [num_blocks, v_heads, v_head_size, qk_head_size]
    const auto state_off = [=](int32_t block, int32_t h, int32_t k_idx, int32_t v_idx) {
        return ((block * v_heads + h) * v_head_size + v_idx) * qk_head_size + k_idx;
    };

    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t token_begin = subsequence_begins[seq];
        const int32_t token_end = subsequence_begins[seq + 1];
        const int32_t block_begin = block_indices_begins[seq];
        const int32_t block_end = block_indices_begins[seq + 1];
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        const int32_t seq_interval = cache_interval[seq];

        for (int32_t h = 0; h < v_heads; h++) {
            const int32_t hk = h / group_size;
            std::vector<float> state(static_cast<size_t>(qk_head_size) * v_head_size, 0.0f);

            const int32_t block_id = block_indices[block_begin];
            for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                    state[k_idx * v_head_size + v_idx] = recurrent_state_table[state_off(block_id, h, k_idx, v_idx)];
                }
            }

            for (int32_t token = token_begin; token < token_end; token++) {
                const auto q_ptr = query.data() + (token * qk_heads + hk) * qk_head_size;
                const auto k_ptr = key.data() + (token * qk_heads + hk) * qk_head_size;

                std::vector<float> q_norm;
                std::vector<float> k_norm;
                normalize_and_scale(q_ptr, qk_head_size, 1.0f / std::sqrt(static_cast<float>(qk_head_size)), q_norm);
                normalize_and_scale(k_ptr, qk_head_size, 1.0f, k_norm);

                const float b_g = std::exp(gate[token * v_heads + h]);
                const float b_beta = beta[token * v_heads + h];

                for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                    const float b_v = value[(token * v_heads + h) * v_head_size + v_idx];

                    float h_k = 0.0f;
                    for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                        auto& s = state[k_idx * v_head_size + v_idx];
                        s *= b_g;
                        h_k += s * k_norm[k_idx];
                    }

                    const float update = (b_v - h_k) * b_beta;
                    float out_v = 0.0f;
                    for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                        auto& s = state[k_idx * v_head_size + v_idx];
                        s += k_norm[k_idx] * update;
                        out_v += s * q_norm[k_idx];
                    }

                    output[(token * v_heads + h) * v_head_size + v_idx] = out_v;
                }

                const int32_t local_token_idx = token - token_begin;
                const int32_t processed_tokens = local_token_idx + 1;
                const bool interval_hit = (seq_interval > 0) && ((processed_tokens % seq_interval) == 0);
                const bool is_last_token = (token == token_end - 1);
                const bool should_store = interval_hit || is_last_token;
                if (should_store) {
                    // When interval==0, keep periodic checkpointing disabled but still
                    // flush state at sequence end to the last available writable slot.
                    const int32_t slot =
                        (seq_interval > 0) ? ((processed_tokens + seq_interval - 1) / seq_interval) : (seq_blocks - 1);
                    OPENVINO_ASSERT(slot >= 0 && slot < seq_blocks);
                    const int32_t block_id = block_indices[block_begin + slot];
                    for (int32_t k_idx = 0; k_idx < qk_head_size; k_idx++) {
                        for (int32_t v_idx = 0; v_idx < v_head_size; v_idx++) {
                            recurrent_state_table[state_off(block_id, h, k_idx, v_idx)] =
                                state[k_idx * v_head_size + v_idx];
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
std::vector<T> tensor_to_vector(const ov::Tensor& tensor) {
    const auto* ptr = tensor.data<const T>();
    return std::vector<T>(ptr, ptr + tensor.get_size());
}

ov::Tensor make_i32_tensor(const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, ov::Shape{values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    return tensor;
}

}  // namespace

namespace ov::test {

std::string PagedGatedDeltaNetLayerTest::getTestCaseName(const testing::TestParamInfo<PagedGatedDeltaNetLayerParams>& obj) {
    const auto& [qk_heads,
                 v_heads,
                 qk_head_size,
                 v_head_size,
                 seq_lengths,
                 cache_intervals,
                 element_type,
                 target_device] = obj.param;
    std::ostringstream result;
    result << "QKHeads=" << qk_heads;
    result << "_VHeads=" << v_heads;
    result << "_QKHeadSize=" << qk_head_size;
    result << "_VHeadSize=" << v_head_size;
    result << "_SeqLens=";
    for (size_t i = 0; i < seq_lengths.size(); i++) {
        if (i > 0)
            result << "x";
        result << seq_lengths[i];
    }
    result << "_Intervals=";
    for (size_t i = 0; i < cache_intervals.size(); i++) {
        if (i > 0)
            result << "x";
        result << cache_intervals[i];
    }
    result << "_Type=" << element_type;
    result << "_Target=" << target_device;
    return result.str();
}

void PagedGatedDeltaNetLayerTest::SetUp() {
    const auto& [qk_heads,
                 v_heads,
                 qk_head_size,
                 v_head_size,
                 seq_lengths,
                 cache_intervals,
                 element_type,
                 device] = GetParam();

    targetDevice = device;

    OPENVINO_ASSERT(!seq_lengths.empty());
    OPENVINO_ASSERT(seq_lengths.size() == cache_intervals.size());

    const int32_t tokens = std::accumulate(seq_lengths.begin(), seq_lengths.end(), 0);
    const int32_t num_sequences = static_cast<int32_t>(seq_lengths.size());

    int32_t num_blocks = 0;
    for (size_t i = 0; i < seq_lengths.size(); i++) {
        OPENVINO_ASSERT(cache_intervals[i] >= 0);
        if (cache_intervals[i] == 0) {
            // interval == 0: use exactly 2 blocks per sequence
            // block 0 for read, block 1 for final write.
            num_blocks += 2;
        } else {
            num_blocks += 1 + (seq_lengths[i] + cache_intervals[i] - 1) / cache_intervals[i];
        }
    }

    const ov::Shape q_shape{static_cast<size_t>(tokens), static_cast<size_t>(qk_heads), static_cast<size_t>(qk_head_size)};
    const ov::Shape v_shape{static_cast<size_t>(tokens), static_cast<size_t>(v_heads), static_cast<size_t>(v_head_size)};
    const ov::Shape state_shape{static_cast<size_t>(num_blocks),
                                static_cast<size_t>(v_heads),
                                static_cast<size_t>(v_head_size),
                                static_cast<size_t>(qk_head_size)};
    const ov::Shape gv_shape{static_cast<size_t>(tokens), static_cast<size_t>(v_heads)};

    init_input_shapes(static_shapes_to_test_representation({q_shape,
                                                            q_shape,
                                                            v_shape,
                                                            state_shape,
                                                            gv_shape,
                                                            gv_shape,
                                                            ov::Shape{static_cast<size_t>(num_sequences + 1)},
                                                            ov::Shape{static_cast<size_t>(num_blocks)},
                                                            ov::Shape{static_cast<size_t>(num_sequences + 1)},
                                                            ov::Shape{static_cast<size_t>(num_sequences)},
                                                            ov::Shape{static_cast<size_t>(num_sequences)}}));

    auto p_query = std::make_shared<ov::op::v0::Parameter>(element_type, q_shape);
    auto p_key = std::make_shared<ov::op::v0::Parameter>(element_type, q_shape);
    auto p_value = std::make_shared<ov::op::v0::Parameter>(element_type, v_shape);
    auto p_state = std::make_shared<ov::op::v0::Parameter>(element_type, state_shape);
    auto p_gate = std::make_shared<ov::op::v0::Parameter>(element_type, gv_shape);
    auto p_beta = std::make_shared<ov::op::v0::Parameter>(element_type, gv_shape);
    auto p_subseq =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(num_sequences + 1)});
    auto p_blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(num_blocks)});
    auto p_block_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(num_sequences + 1)});
    auto p_past_lens =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(num_sequences)});
    auto p_cache_interval =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{static_cast<size_t>(num_sequences)});

    auto pgdn = std::make_shared<ov::op::internal::PagedGatedDeltaNet>(p_query,
                                                                        p_key,
                                                                        p_value,
                                                                        p_state,
                                                                        p_gate,
                                                                        p_beta,
                                                                        p_subseq,
                                                                        p_blocks,
                                                                        p_block_begins,
                                                                        p_past_lens,
                                                                        p_cache_interval,
                                                                        true,
                                                                        1e-6f,
                                                                        1e-6f);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(pgdn)},
                                           ov::ParameterVector{p_query,
                                                               p_key,
                                                               p_value,
                                                               p_state,
                                                               p_gate,
                                                               p_beta,
                                                               p_subseq,
                                                               p_blocks,
                                                               p_block_begins,
                                                               p_past_lens,
                                                               p_cache_interval});
}

void PagedGatedDeltaNetLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();

    const auto& [qk_heads,
                 v_heads,
                 qk_head_size,
                 v_head_size,
                 seq_lengths,
                 cache_intervals,
                 element_type,
                 device] = GetParam();
    const auto num_sequences = static_cast<int32_t>(seq_lengths.size());

    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<int32_t> past_lens;
    std::vector<int32_t> cache_interval;

    subsequence_begins.reserve(static_cast<size_t>(num_sequences + 1));
    block_indices_begins.reserve(static_cast<size_t>(num_sequences + 1));
    past_lens.reserve(static_cast<size_t>(num_sequences));
    cache_interval.reserve(static_cast<size_t>(num_sequences));

    subsequence_begins.push_back(0);
    block_indices_begins.push_back(0);

    int32_t total_blocks = 0;
    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t seq_len = seq_lengths[seq];
        const int32_t seq_interval = cache_intervals[seq];
        OPENVINO_ASSERT(seq_interval >= 0);

        subsequence_begins.push_back(subsequence_begins.back() + seq_len);
        past_lens.push_back(1 + (seq % 3));
        cache_interval.push_back(seq_interval);

        const int32_t required_slots =
            (seq_interval == 0) ? 2 : (1 + (seq_len + seq_interval - 1) / seq_interval);
        for (int32_t i = 0; i < required_slots; i++) {
            block_indices.push_back(total_blocks + i);
        }
        total_blocks += required_slots;
        block_indices_begins.push_back(total_blocks);
    }

    const auto& params = function->get_parameters();
    for (size_t i = 0; i < params.size(); i++) {
        const auto& param = params[i];
        const auto& shape = targetInputStaticShapes[i];

        if (i == 4) {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
        } else if (i == 5) {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(0, 1, 1000, 1));
        } else if (i <= 3) {
            inputs[param] = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                    shape,
                                                                    ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
        } else if (i == 6) {
            inputs[param] = make_i32_tensor(subsequence_begins);
        } else if (i == 7) {
            inputs[param] = make_i32_tensor(block_indices);
        } else if (i == 8) {
            inputs[param] = make_i32_tensor(block_indices_begins);
        } else if (i == 9) {
            inputs[param] = make_i32_tensor(past_lens);
        } else if (i == 10) {
            inputs[param] = make_i32_tensor(cache_interval);
        }
    }

    OPENVINO_ASSERT(qk_heads > 0 && v_heads > 0 && qk_head_size > 0 && v_head_size > 0);
}

std::vector<ov::Tensor> PagedGatedDeltaNetLayerTest::calculate_refs() {
    const auto& [qk_heads,
                 v_heads,
                 qk_head_size,
                 v_head_size,
                 seq_lengths,
                 cache_intervals,
                 element_type,
                 device] = GetParam();
    const auto& params = function->get_parameters();

    auto query = tensor_to_vector<float>(inputs.at(params[0]));
    auto key = tensor_to_vector<float>(inputs.at(params[1]));
    auto value = tensor_to_vector<float>(inputs.at(params[2]));
    auto state = tensor_to_vector<float>(inputs.at(params[3]));
    auto gate = tensor_to_vector<float>(inputs.at(params[4]));
    auto beta = tensor_to_vector<float>(inputs.at(params[5]));
    auto subsequence_begins = tensor_to_vector<int32_t>(inputs.at(params[6]));
    auto block_indices = tensor_to_vector<int32_t>(inputs.at(params[7]));
    auto block_indices_begins = tensor_to_vector<int32_t>(inputs.at(params[8]));
    auto past_lens = tensor_to_vector<int32_t>(inputs.at(params[9]));
    auto cache_interval = tensor_to_vector<int32_t>(inputs.at(params[10]));

    std::vector<float> ref_output;
    run_reference(query,
                  key,
                  value,
                  gate,
                  beta,
                  state,
                  subsequence_begins,
                  block_indices,
                  block_indices_begins,
                  past_lens,
                  cache_interval,
                  qk_heads,
                  v_heads,
                  qk_head_size,
                  v_head_size,
                  ref_output);

    ref_recurrent_state_table = state;

    ov::Tensor output_tensor(ov::element::f32, function->output(0).get_shape());
    std::copy(ref_output.begin(), ref_output.end(), output_tensor.data<float>());
    return {output_tensor};
}

void PagedGatedDeltaNetLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                          const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    abs_threshold = 2e-4f;
    rel_threshold = 1e-5f;
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);

    const auto& state_param = function->get_parameters().at(3);
    const auto actual_state_tensor = inferRequest.get_tensor(state_param);
    ov::Tensor expected_state_tensor(ov::element::f32, state_param->get_shape());
    std::copy(ref_recurrent_state_table.begin(),
              ref_recurrent_state_table.end(),
              expected_state_tensor.data<float>());
    ov::test::utils::compare(expected_state_tensor, actual_state_tensor, abs_threshold, rel_threshold);
}

}  // namespace ov::test
