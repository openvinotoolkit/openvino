// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/paged_selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

namespace {

template <typename T>
void run_reference(const std::vector<T>& A,
                   const std::vector<T>& dt,
                   const std::vector<T>& B,
                   const std::vector<T>& x,
                   const std::vector<T>& C,
                   std::vector<T>& recurrent_state_table,
                   const std::vector<int32_t>& subsequence_begins,
                   const std::vector<int32_t>& block_indices,
                   const std::vector<int32_t>& block_indices_begins,
                   const std::vector<int32_t>& num_processed_tokens,
                   const std::vector<int32_t>& cache_interval,
                   int32_t num_heads,
                   int32_t num_groups,
                   int32_t head_dim,
                   int32_t state_size,
                   std::vector<T>& output) {
    const int32_t tokens = static_cast<int32_t>(x.size()) / (num_heads * head_dim);
    const int32_t heads_per_group = num_heads / num_groups;
    const int32_t num_sequences = static_cast<int32_t>(subsequence_begins.size()) - 1;
    output.resize(static_cast<size_t>(tokens) * num_heads * head_dim);

    const auto state_off = [=](int32_t block, int32_t h, int32_t p, int32_t n) {
        return ((block * num_heads + h) * head_dim + p) * state_size + n;
    };

    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t token_begin = subsequence_begins[seq];
        const int32_t token_end = subsequence_begins[seq + 1];
        const int32_t block_begin = block_indices_begins[seq];
        const int32_t block_end = block_indices_begins[seq + 1];
        const int32_t seq_blocks = std::max(block_end - block_begin, 0);
        const int32_t processed = num_processed_tokens[seq];
        const int32_t interval = cache_interval[seq];
        const int32_t prev_nums = interval > 0 ? (processed % interval) : 0;
        const int32_t first_block = block_indices[block_begin];

        for (int32_t h = 0; h < num_heads; h++) {
            const int32_t g = h / heads_per_group;
            std::vector<float> state(static_cast<size_t>(head_dim) * state_size, 0.f);
            for (int32_t p = 0; p < head_dim; p++) {
                for (int32_t n = 0; n < state_size; n++) {
                    state[p * state_size + n] =
                        static_cast<float>(recurrent_state_table[state_off(first_block, h, p, n)]);
                }
            }

            for (int32_t token = token_begin; token < token_end; token++) {
                const float dt_value = static_cast<float>(dt[token * num_heads + h]);
                const float dA = std::exp(static_cast<float>(A[h]) * dt_value);
                for (int32_t p = 0; p < head_dim; p++) {
                    const float x_value = static_cast<float>(x[(token * num_heads + h) * head_dim + p]);
                    float acc = 0.f;
                    for (int32_t n = 0; n < state_size; n++) {
                        float& s = state[p * state_size + n];
                        s = s * dA +
                            x_value * dt_value * static_cast<float>(B[(token * num_groups + g) * state_size + n]);
                        acc += s * static_cast<float>(C[(token * num_groups + g) * state_size + n]);
                    }
                    output[(token * num_heads + h) * head_dim + p] = static_cast<T>(acc);
                }

                const int32_t processed_tokens = (token - token_begin) + 1;
                const int32_t cached_tokens = prev_nums + processed_tokens;
                const bool reached_interval_boundary = interval > 0 && ((cached_tokens % interval) == 0);
                const bool reached_sequence_end = token == token_end - 1;
                if (interval > 0 && (reached_interval_boundary || reached_sequence_end)) {
                    const int32_t slot = 1 + (cached_tokens - 1) / interval;
                    if (slot < seq_blocks) {
                        const int32_t block_id = block_indices[block_begin + slot];
                        for (int32_t p = 0; p < head_dim; p++) {
                            for (int32_t n = 0; n < state_size; n++) {
                                recurrent_state_table[state_off(block_id, h, p, n)] =
                                    static_cast<T>(state[p * state_size + n]);
                            }
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

template <typename T>
std::vector<ov::Tensor> calculate_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                             const std::shared_ptr<ov::Model>& function,
                                             int32_t num_heads,
                                             int32_t num_groups,
                                             int32_t head_dim,
                                             int32_t state_size,
                                             const ov::element::Type& data_type) {
    const auto& params = function->get_parameters();

    auto A = tensor_to_vector<T>(host_inputs.at(params[0]));
    auto dt = tensor_to_vector<T>(host_inputs.at(params[1]));
    auto B = tensor_to_vector<T>(host_inputs.at(params[2]));
    auto x = tensor_to_vector<T>(host_inputs.at(params[3]));
    auto C = tensor_to_vector<T>(host_inputs.at(params[4]));
    auto state = tensor_to_vector<T>(host_inputs.at(params[5]));
    auto subsequence_begins = tensor_to_vector<int32_t>(host_inputs.at(params[6]));
    auto block_indices = tensor_to_vector<int32_t>(host_inputs.at(params[7]));
    auto block_indices_begins = tensor_to_vector<int32_t>(host_inputs.at(params[8]));
    auto num_processed_tokens = tensor_to_vector<int32_t>(host_inputs.at(params[9]));
    auto cache_interval = tensor_to_vector<int32_t>(host_inputs.at(params[10]));

    std::vector<T> ref_output;
    run_reference(A,
                  dt,
                  B,
                  x,
                  C,
                  state,
                  subsequence_begins,
                  block_indices,
                  block_indices_begins,
                  num_processed_tokens,
                  cache_interval,
                  num_heads,
                  num_groups,
                  head_dim,
                  state_size,
                  ref_output);

    ov::Tensor output_tensor(data_type, host_inputs.at(params[3]).get_shape());
    std::copy(ref_output.begin(), ref_output.end(), output_tensor.data<T>());

    ov::Tensor state_tensor(data_type, host_inputs.at(params[5]).get_shape());
    std::copy(state.begin(), state.end(), state_tensor.data<T>());

    return {output_tensor, state_tensor};
}

}  // namespace

namespace ov::test {

std::string PagedSelectiveSSMLayerTest::getTestCaseName(
    const testing::TestParamInfo<PagedSelectiveSSMLayerParams>& obj) {
    const auto& [num_heads,
                 num_groups,
                 head_dim,
                 state_size,
                 seq_lengths,
                 cache_intervals,
                 element_type,
                 target_device] = obj.param;
    std::ostringstream result;
    result << "Heads=" << num_heads;
    result << "_Groups=" << num_groups;
    result << "_HeadDim=" << head_dim;
    result << "_StateSize=" << state_size;
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

void PagedSelectiveSSMLayerTest::SetUp() {
    const auto& [num_heads, num_groups, head_dim, state_size, seq_lengths, cache_intervals, data_type, device] =
        GetParam();
    if (device.find("CPU") != std::string::npos) {
        if ((data_type == ov::element::bf16) && !with_cpu_x86_avx512_core_amx_bf16()) {
            GTEST_SKIP();
        }
        if ((data_type == ov::element::f16) && !with_cpu_x86_avx512_core_fp16()) {
            GTEST_SKIP();
        }
    }

    targetDevice = device;
    this->data_type = data_type;
    configuration[ov::hint::inference_precision.name()] = data_type;
    OPENVINO_ASSERT(!seq_lengths.empty());
    OPENVINO_ASSERT(seq_lengths.size() == cache_intervals.size());

    const int32_t tokens = std::accumulate(seq_lengths.begin(), seq_lengths.end(), 0);
    const int32_t num_sequences = static_cast<int32_t>(seq_lengths.size());

    int32_t num_blocks = 0;
    for (size_t i = 0; i < seq_lengths.size(); i++) {
        OPENVINO_ASSERT(cache_intervals[i] >= 0);
        const int32_t processed = 1 + static_cast<int32_t>(i % 3);
        if (cache_intervals[i] == 0) {
            num_blocks += 2;
        } else {
            const int32_t prev_nums = processed % cache_intervals[i];
            const int32_t write_blocks = (prev_nums + seq_lengths[i] + cache_intervals[i] - 1) / cache_intervals[i];
            num_blocks += 1 + write_blocks;
        }
    }

    const ov::Shape A_shape{static_cast<size_t>(num_heads)};
    const ov::Shape dt_shape{static_cast<size_t>(tokens), static_cast<size_t>(num_heads)};
    const ov::Shape BC_shape{static_cast<size_t>(tokens),
                             static_cast<size_t>(num_groups),
                             static_cast<size_t>(state_size)};
    const ov::Shape x_shape{static_cast<size_t>(tokens), static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)};
    const ov::Shape state_shape{static_cast<size_t>(num_blocks),
                                static_cast<size_t>(num_heads),
                                static_cast<size_t>(head_dim),
                                static_cast<size_t>(state_size)};

    init_input_shapes(static_shapes_to_test_representation({A_shape,
                                                            dt_shape,
                                                            BC_shape,
                                                            x_shape,
                                                            BC_shape,
                                                            state_shape,
                                                            ov::Shape{static_cast<size_t>(num_sequences + 1)},
                                                            ov::Shape{static_cast<size_t>(num_blocks)},
                                                            ov::Shape{static_cast<size_t>(num_sequences + 1)},
                                                            ov::Shape{static_cast<size_t>(num_sequences)},
                                                            ov::Shape{static_cast<size_t>(num_sequences)}}));

    auto p_A = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{num_heads});
    auto p_dt = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads});
    auto p_B = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_groups, state_size});
    auto p_x = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads, head_dim});
    auto p_C = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_groups, state_size});
    auto p_state =
        std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto p_subseq = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto p_blocks = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto p_block_begins = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto p_num_processed = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto p_cache_interval = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::PartialShape{-1});
    auto pssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(p_A,
                                                                      p_dt,
                                                                      p_B,
                                                                      p_x,
                                                                      p_C,
                                                                      p_state,
                                                                      p_subseq,
                                                                      p_blocks,
                                                                      p_block_begins,
                                                                      p_num_processed,
                                                                      p_cache_interval);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(pssm)},
                                           ov::ParameterVector{p_A,
                                                               p_dt,
                                                               p_B,
                                                               p_x,
                                                               p_C,
                                                               p_state,
                                                               p_subseq,
                                                               p_blocks,
                                                               p_block_begins,
                                                               p_num_processed,
                                                               p_cache_interval});
}

void PagedSelectiveSSMLayerTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    const auto& [num_heads, num_groups, head_dim, state_size, seq_lengths, cache_intervals, element_type, device] =
        GetParam();
    const auto num_sequences = static_cast<int32_t>(seq_lengths.size());

    std::vector<int32_t> subsequence_begins;
    std::vector<int32_t> block_indices;
    std::vector<int32_t> block_indices_begins;
    std::vector<int32_t> num_processed_tokens;
    std::vector<int32_t> cache_interval;

    subsequence_begins.reserve(static_cast<size_t>(num_sequences + 1));
    block_indices_begins.reserve(static_cast<size_t>(num_sequences + 1));
    num_processed_tokens.reserve(static_cast<size_t>(num_sequences));
    cache_interval.reserve(static_cast<size_t>(num_sequences));

    subsequence_begins.push_back(0);
    block_indices_begins.push_back(0);

    int32_t total_blocks = 0;
    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t seq_len = seq_lengths[seq];
        const int32_t seq_interval = cache_intervals[seq];
        const int32_t processed = 1 + (seq % 3);

        subsequence_begins.push_back(subsequence_begins.back() + seq_len);
        num_processed_tokens.push_back(processed);
        cache_interval.push_back(seq_interval);

        int32_t required_slots = 2;
        if (seq_interval > 0) {
            const int32_t prev_nums = processed % seq_interval;
            const int32_t write_blocks = (prev_nums + seq_len + seq_interval - 1) / seq_interval;
            required_slots = 1 + write_blocks;
        }
        for (int32_t i = 0; i < required_slots; i++) {
            block_indices.push_back(total_blocks + i);
        }
        total_blocks += required_slots;
        block_indices_begins.push_back(total_blocks);
    }

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

        if (i == 0) {
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                             shape,
                                                             ov::test::utils::InputGenerateData(-0.5f, 0.7f, 1000, 1));
        } else if (i == 1) {
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                             shape,
                                                             ov::test::utils::InputGenerateData(0.0f, 0.5f, 1000, 1));
        } else if (i <= 5) {
            tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(),
                                                             shape,
                                                             ov::test::utils::InputGenerateData(-0.5f, 1.0f, 1000, 1));
        } else if (i == 6) {
            tensor = make_i32_tensor(subsequence_begins);
        } else if (i == 7) {
            tensor = make_i32_tensor(block_indices);
        } else if (i == 8) {
            tensor = make_i32_tensor(block_indices_begins);
        } else if (i == 9) {
            tensor = make_i32_tensor(num_processed_tokens);
        } else if (i == 10) {
            tensor = make_i32_tensor(cache_interval);
        }

        host_inputs[param] = tensor;
        if (use_remote_tensors && i <= 5) {
            auto remote_tensor = remote_context.create_tensor(param->get_element_type(), shape);
            remote_tensor.copy_from(tensor);
            inputs[param] = remote_tensor;
        } else {
            inputs[param] = tensor;
        }
    }
}

std::vector<ov::Tensor> PagedSelectiveSSMLayerTest::calculate_refs() {
    const auto& [num_heads, num_groups, head_dim, state_size, seq_lengths, cache_intervals, element_type, device] =
        GetParam();

    if (element_type == ov::element::f16) {
        return calculate_typed_refs<ov::float16>(host_inputs,
                                                 function,
                                                 num_heads,
                                                 num_groups,
                                                 head_dim,
                                                 state_size,
                                                 element_type);
    }
    if (element_type == ov::element::bf16) {
        return calculate_typed_refs<ov::bfloat16>(host_inputs,
                                                  function,
                                                  num_heads,
                                                  num_groups,
                                                  head_dim,
                                                  state_size,
                                                  element_type);
    }
    return calculate_typed_refs<float>(host_inputs,
                                       function,
                                       num_heads,
                                       num_groups,
                                       head_dim,
                                       state_size,
                                       element_type);
}

std::vector<ov::Tensor> PagedSelectiveSSMLayerTest::get_plugin_outputs() {
    auto outputs = SubgraphBaseTest::get_plugin_outputs();

    const auto& state_param = function->get_parameters().at(5);
    const auto actual_state_tensor = inferRequest.get_tensor(state_param);
    ov::Tensor host_state_tensor(actual_state_tensor.get_element_type(), actual_state_tensor.get_shape());
    actual_state_tensor.copy_to(host_state_tensor);
    outputs.push_back(host_state_tensor);

    return outputs;
}

void PagedSelectiveSSMLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    if (data_type == ov::element::bf16) {
        abs_threshold = 1e-3f;
        rel_threshold = 1e-2f;
    } else if (data_type == ov::element::f16) {
        abs_threshold = 5e-4f;
        rel_threshold = 1e-5f;
    } else {
        abs_threshold = 2e-4f;
        rel_threshold = 1e-5f;
    }
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
}

}  // namespace ov::test
