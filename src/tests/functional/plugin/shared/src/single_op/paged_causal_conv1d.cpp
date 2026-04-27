// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_causal_conv1d.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

struct ShapeConfig {
    int32_t tokens;
    int32_t num_sequences;
    int32_t num_blocks;
};

ShapeConfig compute_shape_config(const std::vector<int32_t>& seq_lengths, const std::vector<int32_t>& cache_intervals) {
    const int32_t tokens = std::accumulate(seq_lengths.begin(), seq_lengths.end(), 0);
    const int32_t num_sequences = static_cast<int32_t>(seq_lengths.size());

    int32_t num_blocks = 0;
    for (size_t i = 0; i < seq_lengths.size(); i++) {
        const int32_t past_len = 1 + static_cast<int32_t>(i % 3);
        if (cache_intervals[i] == 0) {
            num_blocks += 2;
        } else {
            const int32_t prev_nums = past_len % cache_intervals[i];
            const int32_t write_blocks = (prev_nums + seq_lengths[i] + cache_intervals[i] - 1) / cache_intervals[i];
            num_blocks += 1 + write_blocks;
        }
    }
    return {tokens, num_sequences, num_blocks};
}

template <typename T>
void run_reference(const std::vector<T>& input_embeds,
                   std::vector<T>& conv_state_table,
                   const std::vector<T>& conv_weight,
                   const std::vector<T>& conv_bias,
                   bool has_bias,
                   const std::vector<int32_t>& subsequence_begins,
                   const std::vector<int32_t>& block_indices,
                   const std::vector<int32_t>& block_indices_begins,
                   const std::vector<int32_t>& past_lens,
                   const std::vector<int32_t>& cache_interval,
                   int32_t hidden_size,
                   int32_t kernel_size,
                   std::vector<T>& output) {
    const size_t state_stride = static_cast<size_t>(hidden_size) * kernel_size;
    const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
    const size_t total_tokens = input_embeds.size() / hidden_size;
    output.resize(total_tokens * hidden_size);

    std::vector<float> local_state(state_stride);

    for (int32_t s = 0; s < num_sequences; s++) {
        const int32_t token_begin = subsequence_begins[s];
        const int32_t token_end = subsequence_begins[s + 1];
        const int32_t blk_begin = block_indices_begins[s];
        const int32_t blk_end = block_indices_begins[s + 1];
        const int32_t block_span = blk_end - blk_begin;
        if (block_span <= 1)
            continue;

        const int32_t seq_interval = cache_interval[s];
        const int32_t prev_nums = (seq_interval > 0) ? (past_lens[s] % seq_interval) : 0;
        const int32_t seq_tokens = token_end - token_begin;

        const int32_t read_block = block_indices[blk_begin];
        for (size_t i = 0; i < state_stride; i++) {
            local_state[i] = static_cast<float>(conv_state_table[static_cast<size_t>(read_block) * state_stride + i]);
        }

        for (int32_t t = 0; t < seq_tokens; t++) {
            const size_t token_idx = static_cast<size_t>(token_begin + t);

            for (int32_t h = 0; h < hidden_size; h++) {
                float* state_h = local_state.data() + static_cast<size_t>(h) * kernel_size;
                for (int32_t k = 0; k + 1 < kernel_size; k++) {
                    state_h[k] = state_h[k + 1];
                }
                state_h[kernel_size - 1] = static_cast<float>(input_embeds[token_idx * hidden_size + h]);

                const size_t weight_off = static_cast<size_t>(h) * kernel_size;
                float sum = has_bias ? static_cast<float>(conv_bias[h]) : 0.0f;
                for (int32_t k = 0; k < kernel_size; k++) {
                    sum += state_h[k] * static_cast<float>(conv_weight[weight_off + k]);
                }
                output[token_idx * hidden_size + h] = static_cast<T>(sum);
            }

            const int32_t cached_tokens = prev_nums + (t + 1);
            const bool interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
            const bool is_last_token = (t == seq_tokens - 1);
            if (interval_hit || is_last_token) {
                const int32_t slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                if (slot < block_span) {
                    const int32_t phys_block = block_indices[blk_begin + slot];
                    for (size_t i = 0; i < state_stride; i++) {
                        conv_state_table[static_cast<size_t>(phys_block) * state_stride + i] =
                            static_cast<T>(local_state[i]);
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

template <typename T>
std::vector<ov::Tensor> calculate_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                             const std::shared_ptr<ov::Model>& function,
                                             int32_t hidden_size,
                                             int32_t kernel_size,
                                             bool has_bias,
                                             const ov::element::Type& data_type) {
    const auto& params = function->get_parameters();

    auto input_embeds = tensor_to_vector<T>(host_inputs.at(params[0]));
    auto state = tensor_to_vector<T>(host_inputs.at(params[1]));
    auto weight = tensor_to_vector<T>(host_inputs.at(params[2]));
    auto bias = tensor_to_vector<T>(host_inputs.at(params[3]));
    auto subsequence_begins = tensor_to_vector<int32_t>(host_inputs.at(params[4]));
    auto block_indices = tensor_to_vector<int32_t>(host_inputs.at(params[5]));
    auto block_indices_begins = tensor_to_vector<int32_t>(host_inputs.at(params[6]));
    auto past_lens = tensor_to_vector<int32_t>(host_inputs.at(params[7]));
    auto cache_interval = tensor_to_vector<int32_t>(host_inputs.at(params[8]));

    std::vector<T> ref_output;
    run_reference(input_embeds,
                  state,
                  weight,
                  bias,
                  has_bias,
                  subsequence_begins,
                  block_indices,
                  block_indices_begins,
                  past_lens,
                  cache_interval,
                  hidden_size,
                  kernel_size,
                  ref_output);

    ov::Tensor output_tensor(data_type, host_inputs.at(params[0]).get_shape());
    std::copy(ref_output.begin(), ref_output.end(), output_tensor.data<T>());

    ov::Tensor state_tensor(data_type, host_inputs.at(params[1]).get_shape());
    std::copy(state.begin(), state.end(), state_tensor.data<T>());

    return {output_tensor, state_tensor};
}

}  // namespace

namespace ov::test {

std::string PagedCausalConv1DLayerTest::getTestCaseName(
    const testing::TestParamInfo<PagedCausalConv1DLayerParams>& obj) {
    const auto& p = obj.param;
    std::ostringstream result;
    result << "Hidden=" << p.hidden_size;
    result << "_Kernel=" << p.kernel_size;
    result << "_Bias=" << (p.has_bias ? "yes" : "no");
    for (size_t s = 0; s < p.seq_lengths_sets.size(); s++) {
        result << "_SeqSet" << s << "=";
        for (size_t i = 0; i < p.seq_lengths_sets[s].size(); i++) {
            if (i > 0)
                result << "x";
            result << p.seq_lengths_sets[s][i];
        }
    }
    for (size_t s = 0; s < p.cache_intervals_sets.size(); s++) {
        result << "_IntSet" << s << "=";
        for (size_t i = 0; i < p.cache_intervals_sets[s].size(); i++) {
            if (i > 0)
                result << "x";
            result << p.cache_intervals_sets[s][i];
        }
    }
    result << "_Type=" << p.element_type;
    result << "_Target=" << p.target_device;
    return result.str();
}

void PagedCausalConv1DLayerTest::SetUp() {
    const auto& p = GetParam();
    targetDevice = p.target_device;
    data_type = p.element_type;
    configuration[ov::hint::inference_precision.name()] = p.element_type;

    const size_t num_iters = p.seq_lengths_sets.size();
    OPENVINO_ASSERT(num_iters > 0);
    OPENVINO_ASSERT(num_iters == p.cache_intervals_sets.size());

    m_iteration_data.resize(num_iters);
    m_current_iteration = 0;

    const auto hidden = static_cast<size_t>(p.hidden_size);
    const auto kernel = static_cast<size_t>(p.kernel_size);
    const ov::Shape weight_shape{hidden, 1, kernel};
    const ov::Shape bias_shape = p.has_bias ? ov::Shape{hidden} : ov::Shape{0};

    // Build per-iteration target shapes and metadata
    std::vector<ov::Shape> embeds_targets;
    std::vector<ov::Shape> state_targets;
    std::vector<ov::Shape> subseq_targets;
    std::vector<ov::Shape> blocks_targets;
    std::vector<ov::Shape> block_begins_targets;
    std::vector<ov::Shape> past_lens_targets;
    std::vector<ov::Shape> interval_targets;

    for (size_t iter = 0; iter < num_iters; iter++) {
        const auto& sl = p.seq_lengths_sets[iter];
        const auto& ci = p.cache_intervals_sets[iter];
        OPENVINO_ASSERT(sl.size() == ci.size());

        const auto cfg = compute_shape_config(sl, ci);

        embeds_targets.push_back({static_cast<size_t>(cfg.tokens), hidden});
        state_targets.push_back({static_cast<size_t>(cfg.num_blocks), hidden, kernel});
        subseq_targets.push_back({static_cast<size_t>(cfg.num_sequences + 1)});
        blocks_targets.push_back({static_cast<size_t>(cfg.num_blocks)});
        block_begins_targets.push_back({static_cast<size_t>(cfg.num_sequences + 1)});
        past_lens_targets.push_back({static_cast<size_t>(cfg.num_sequences)});
        interval_targets.push_back({static_cast<size_t>(cfg.num_sequences)});

        // Precompute metadata for generate_inputs
        auto& data = m_iteration_data[iter];
        const int32_t num_sequences = static_cast<int32_t>(sl.size());

        data.subsequence_begins.clear();
        data.block_indices.clear();
        data.block_indices_begins.clear();
        data.past_lens.clear();
        data.cache_interval.clear();

        data.subsequence_begins.push_back(0);
        data.block_indices_begins.push_back(0);

        int32_t total_blocks = 0;
        for (int32_t seq = 0; seq < num_sequences; seq++) {
            const int32_t seq_len = sl[seq];
            const int32_t seq_interval = ci[seq];
            const int32_t seq_past_len = 1 + (seq % 3);

            data.subsequence_begins.push_back(data.subsequence_begins.back() + seq_len);
            data.past_lens.push_back(seq_past_len);
            data.cache_interval.push_back(seq_interval);

            int32_t required_slots = 2;
            if (seq_interval > 0) {
                const int32_t prev_nums = seq_past_len % seq_interval;
                const int32_t write_blocks = (prev_nums + seq_len + seq_interval - 1) / seq_interval;
                required_slots = 1 + write_blocks;
            }
            for (int32_t i = 0; i < required_slots; i++) {
                data.block_indices.push_back(total_blocks + i);
            }
            total_blocks += required_slots;
            data.block_indices_begins.push_back(total_blocks);
        }
    }

    // Use dynamic partial shapes for dimensions that vary across iterations
    init_input_shapes({
        InputShape{ov::PartialShape{-1, static_cast<int64_t>(hidden)}, embeds_targets},
        InputShape{ov::PartialShape{-1, static_cast<int64_t>(hidden), static_cast<int64_t>(kernel)}, state_targets},
        InputShape{ov::PartialShape{static_cast<int64_t>(hidden), 1, static_cast<int64_t>(kernel)}, {weight_shape}},
        InputShape{ov::PartialShape{static_cast<int64_t>(p.has_bias ? p.hidden_size : 0)}, {bias_shape}},
        InputShape{ov::PartialShape{-1}, subseq_targets},
        InputShape{ov::PartialShape{-1}, blocks_targets},
        InputShape{ov::PartialShape{-1}, block_begins_targets},
        InputShape{ov::PartialShape{-1}, past_lens_targets},
        InputShape{ov::PartialShape{-1}, interval_targets},
    });

    // Build model with dynamic shapes
    auto p_embeds = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[0]);
    auto p_state = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[1]);
    auto p_weight = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[2]);
    auto p_bias = std::make_shared<ov::op::v0::Parameter>(data_type, inputDynamicShapes[3]);
    auto p_subseq = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[4]);
    auto p_blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[5]);
    auto p_block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[6]);
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[7]);
    auto p_cache_interval = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, inputDynamicShapes[8]);

    auto conv1d = std::make_shared<ov::op::internal::PagedCausalConv1D>(p_embeds,
                                                                        p_state,
                                                                        p_weight,
                                                                        p_bias,
                                                                        p_subseq,
                                                                        p_blocks,
                                                                        p_block_begins,
                                                                        p_past_lens,
                                                                        p_cache_interval);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(conv1d)},
                                           ov::ParameterVector{p_embeds,
                                                               p_state,
                                                               p_weight,
                                                               p_bias,
                                                               p_subseq,
                                                               p_blocks,
                                                               p_block_begins,
                                                               p_past_lens,
                                                               p_cache_interval});
}

void PagedCausalConv1DLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    const auto& p = GetParam();
    const size_t iter = m_current_iteration;
    m_current_iteration = (m_current_iteration + 1) % m_iteration_data.size();
    const auto& iter_data = m_iteration_data[iter];

    const auto& params = function->get_parameters();
    const bool use_remote_tensors = targetDevice == "GPU";
    ov::RemoteContext remote_context;
    if (use_remote_tensors) {
        remote_context = compiledModel.get_context();
    }

    for (size_t i = 0; i < params.size(); i++) {
        const auto& param = params[i];
        const auto& shape = targetInputStaticShapes[i];
        ov::Tensor tensor;

        if (i <= 1) {
            // input_embeds and conv_state_table
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                             shape,
                                                             ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
        } else if (i == 2) {
            // conv_weight
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                             shape,
                                                             ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
        } else if (i == 3) {
            // conv_bias
            if (p.has_bias) {
                tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                                 shape,
                                                                 ov::test::utils::InputGenerateData(-1, 1, 1000, 1));
            } else {
                tensor = ov::Tensor(param->get_element_type(), shape);
            }
        } else if (i == 4) {
            tensor = make_i32_tensor(iter_data.subsequence_begins);
        } else if (i == 5) {
            tensor = make_i32_tensor(iter_data.block_indices);
        } else if (i == 6) {
            tensor = make_i32_tensor(iter_data.block_indices_begins);
        } else if (i == 7) {
            tensor = make_i32_tensor(iter_data.past_lens);
        } else if (i == 8) {
            tensor = make_i32_tensor(iter_data.cache_interval);
        }

        host_inputs[param] = tensor;

        if (use_remote_tensors && i <= 3) {
            auto remote_tensor = remote_context.create_tensor(param->get_element_type(), shape);
            remote_tensor.copy_from(tensor);
            inputs[param] = remote_tensor;
        } else {
            inputs[param] = tensor;
        }
    }
}

std::vector<ov::Tensor> PagedCausalConv1DLayerTest::calculate_refs() {
    const auto& p = GetParam();

    if (data_type == ov::element::f16) {
        return calculate_typed_refs<ov::float16>(host_inputs,
                                                 function,
                                                 p.hidden_size,
                                                 p.kernel_size,
                                                 p.has_bias,
                                                 data_type);
    }

    if (data_type == ov::element::bf16) {
        return calculate_typed_refs<ov::bfloat16>(host_inputs,
                                                  function,
                                                  p.hidden_size,
                                                  p.kernel_size,
                                                  p.has_bias,
                                                  data_type);
    }

    return calculate_typed_refs<float>(host_inputs, function, p.hidden_size, p.kernel_size, p.has_bias, data_type);
}

std::vector<ov::Tensor> PagedCausalConv1DLayerTest::get_plugin_outputs() {
    auto outputs = SubgraphBaseTest::get_plugin_outputs();

    // Read back the state table (input 1) which was modified in-place
    const auto& state_param = function->get_parameters().at(1);
    const auto actual_state_tensor = inferRequest.get_tensor(state_param);
    ov::Tensor host_state_tensor(actual_state_tensor.get_element_type(), actual_state_tensor.get_shape());
    actual_state_tensor.copy_to(host_state_tensor);
    outputs.push_back(host_state_tensor);

    return outputs;
}

void PagedCausalConv1DLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    if (data_type == ov::element::bf16) {
        abs_threshold = 1e-2f;
        rel_threshold = 1e-2f;
    } else if (data_type == ov::element::f16) {
        abs_threshold = 1e-3f;
        rel_threshold = 1e-3f;
    } else {
        abs_threshold = 1e-5f;
        rel_threshold = 1e-5f;
    }
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
}

}  // namespace ov::test
