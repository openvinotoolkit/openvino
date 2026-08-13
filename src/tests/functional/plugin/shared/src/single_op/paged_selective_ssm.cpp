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

template <typename T, typename IndexT>
void run_reference(const std::vector<T>& A,
                   const std::vector<T>& dt,
                   const std::vector<T>& B,
                   const std::vector<T>& x,
                   const std::vector<T>& C,
                   std::vector<T>& recurrent_state_table,
                   const std::vector<IndexT>& subsequence_begins,
                   const std::vector<IndexT>& block_indices,
                   const std::vector<IndexT>& block_indices_begins,
                   const std::vector<IndexT>& num_processed_tokens,
                   const std::vector<IndexT>& cache_interval,
                   size_t num_heads,
                   size_t num_groups,
                   size_t head_dim,
                   size_t state_size,
                   std::vector<T>& output) {
    const size_t tokens = x.size() / (num_heads * head_dim);
    const size_t heads_per_group = num_heads / num_groups;
    const size_t num_sequences = subsequence_begins.size() - 1;
    output.resize(tokens * num_heads * head_dim);

    const auto state_off = [=](size_t block, size_t head, size_t position, size_t state) {
        return ((block * num_heads + head) * head_dim + position) * state_size + state;
    };

    for (size_t sequence = 0; sequence < num_sequences; ++sequence) {
        const auto token_begin = static_cast<size_t>(subsequence_begins[sequence]);
        const auto token_end = static_cast<size_t>(subsequence_begins[sequence + 1]);
        if (token_begin == token_end) {
            continue;
        }

        const auto logical_block_begin = static_cast<size_t>(block_indices_begins[sequence]);
        const auto logical_block_end = static_cast<size_t>(block_indices_begins[sequence + 1]);
        const auto interval = cache_interval[sequence];
        const auto read_block = static_cast<size_t>(block_indices[logical_block_begin]);

        for (size_t head = 0; head < num_heads; ++head) {
            const size_t group = head / heads_per_group;
            std::vector<float> state(head_dim * state_size);
            for (size_t position = 0; position < head_dim; ++position) {
                for (size_t state_index = 0; state_index < state_size; ++state_index) {
                    state[position * state_size + state_index] =
                        static_cast<float>(recurrent_state_table[state_off(read_block, head, position, state_index)]);
                }
            }

            for (size_t token = token_begin; token < token_end; ++token) {
                const float delta = static_cast<float>(dt[token * num_heads + head]);
                const float decay = std::exp(static_cast<float>(A[head]) * delta);
                const auto projection_base = (token * num_groups + group) * state_size;
                std::vector<float> input_projection(state_size);
                for (size_t state_index = 0; state_index < state_size; ++state_index) {
                    input_projection[state_index] =
                        delta * static_cast<float>(B[projection_base + state_index]);
                }

                for (size_t position = 0; position < head_dim; ++position) {
                    const auto state_base = position * state_size;
                    const float input = static_cast<float>(x[(token * num_heads + head) * head_dim + position]);
                    for (size_t state_index = 0; state_index < state_size; ++state_index) {
                        auto& state_value = state[state_base + state_index];
                        state_value = state_value * decay + input * input_projection[state_index];
                    }

                    double reduction = 0.0;
                    for (size_t state_index = 0; state_index < state_size; ++state_index) {
                        reduction += static_cast<double>(state[state_base + state_index]) *
                                     static_cast<float>(C[projection_base + state_index]);
                    }
                    output[(token * num_heads + head) * head_dim + position] = static_cast<T>(reduction);
                }

                if (interval > 0) {
                    const auto positive_interval = static_cast<uint64_t>(interval);
                    const auto previous_offset =
                        static_cast<uint64_t>(num_processed_tokens[sequence]) % positive_interval;
                    const auto current_tokens = token - token_begin + 1;
                    const auto cached_tokens = previous_offset + current_tokens;
                    const bool is_boundary = cached_tokens % positive_interval == 0;
                    const bool is_last = token + 1 == token_end;
                    if (is_boundary || is_last) {
                        const auto write_slot = 1 + (cached_tokens - 1) / positive_interval;
                        OPENVINO_ASSERT(logical_block_begin + write_slot < logical_block_end);
                        const auto write_block =
                            static_cast<size_t>(block_indices[logical_block_begin + write_slot]);
                        for (size_t position = 0; position < head_dim; ++position) {
                            for (size_t state_index = 0; state_index < state_size; ++state_index) {
                                recurrent_state_table[state_off(write_block, head, position, state_index)] =
                                    static_cast<T>(state[position * state_size + state_index]);
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

ov::Tensor make_index_tensor(const std::vector<int32_t>& values, const ov::element::Type& index_type) {
    ov::Tensor tensor(index_type, ov::Shape{values.size()});
    if (index_type == ov::element::i32) {
        std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    } else {
        std::transform(values.begin(), values.end(), tensor.data<int64_t>(), [](int32_t value) {
            return static_cast<int64_t>(value);
        });
    }
    return tensor;
}

template <typename T, typename IndexT>
std::vector<ov::Tensor> calculate_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                             const std::shared_ptr<ov::Model>& function,
                                             size_t num_heads,
                                             size_t num_groups,
                                             size_t head_dim,
                                             size_t state_size,
                                             const ov::element::Type& data_type) {
    const auto& params = function->get_parameters();

    auto A = tensor_to_vector<T>(host_inputs.at(params[0]));
    auto dt = tensor_to_vector<T>(host_inputs.at(params[1]));
    auto B = tensor_to_vector<T>(host_inputs.at(params[2]));
    auto x = tensor_to_vector<T>(host_inputs.at(params[3]));
    auto C = tensor_to_vector<T>(host_inputs.at(params[4]));
    auto state = tensor_to_vector<T>(host_inputs.at(params[5]));
    auto subsequence_begins = tensor_to_vector<IndexT>(host_inputs.at(params[6]));
    auto block_indices = tensor_to_vector<IndexT>(host_inputs.at(params[7]));
    auto block_indices_begins = tensor_to_vector<IndexT>(host_inputs.at(params[8]));
    auto num_processed_tokens = tensor_to_vector<IndexT>(host_inputs.at(params[9]));
    auto cache_interval = tensor_to_vector<IndexT>(host_inputs.at(params[10]));

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

template <typename T>
std::vector<ov::Tensor> calculate_data_typed_refs(
    const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
    const std::shared_ptr<ov::Model>& function,
    size_t num_heads,
    size_t num_groups,
    size_t head_dim,
    size_t state_size,
    const ov::element::Type& data_type,
    const ov::element::Type& index_type) {
    if (index_type == ov::element::i64) {
        return calculate_typed_refs<T, int64_t>(
            host_inputs, function, num_heads, num_groups, head_dim, state_size, data_type);
    }
    return calculate_typed_refs<T, int32_t>(
        host_inputs, function, num_heads, num_groups, head_dim, state_size, data_type);
}

int32_t count_logical_blocks(const std::vector<int32_t>& sequence_lengths,
                             const std::vector<int32_t>& processed_tokens,
                             const std::vector<int32_t>& cache_intervals) {
    OPENVINO_ASSERT(sequence_lengths.size() == processed_tokens.size());
    OPENVINO_ASSERT(sequence_lengths.size() == cache_intervals.size());

    int32_t result = 0;
    for (size_t sequence = 0; sequence < sequence_lengths.size(); ++sequence) {
        const auto sequence_length = sequence_lengths[sequence];
        if (sequence_length == 0) {
            continue;
        }

        const auto interval = cache_intervals[sequence];
        if (interval <= 0) {
            ++result;
            continue;
        }

        const auto previous_offset = processed_tokens[sequence] % interval;
        const auto write_count = (previous_offset + sequence_length + interval - 1) / interval;
        result += 1 + write_count;
    }
    return result;
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
                 num_processed_tokens,
                 cache_intervals,
                 element_type,
                 index_type,
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
    result << "_Processed=";
    for (size_t i = 0; i < num_processed_tokens.size(); i++) {
        if (i > 0)
            result << "x";
        result << num_processed_tokens[i];
    }
    result << "_Intervals=";
    for (size_t i = 0; i < cache_intervals.size(); i++) {
        if (i > 0)
            result << "x";
        if (cache_intervals[i] < 0) {
            result << "neg" << -cache_intervals[i];
        } else {
            result << cache_intervals[i];
        }
    }
    result << "_Type=" << element_type;
    result << "_IndexType=" << index_type;
    result << "_Target=" << target_device;
    return result.str();
}

void PagedSelectiveSSMLayerTest::SetUp() {
    const auto& [num_heads,
                 num_groups,
                 head_dim,
                 state_size,
                 seq_lengths,
                 num_processed_tokens,
                 cache_intervals,
                 data_type,
                 index_type,
                 device] = GetParam();
    targetDevice = device;
    this->data_type = data_type;
    configuration[ov::hint::inference_precision.name()] = data_type;
    OPENVINO_ASSERT(!seq_lengths.empty());
    OPENVINO_ASSERT(seq_lengths.size() == num_processed_tokens.size());
    OPENVINO_ASSERT(seq_lengths.size() == cache_intervals.size());

    auto alternate_seq_lengths = seq_lengths;
    auto alternate_processed_tokens = num_processed_tokens;
    auto alternate_cache_intervals = cache_intervals;
    alternate_seq_lengths.push_back(1);
    alternate_processed_tokens.push_back(0);
    alternate_cache_intervals.push_back(2);

    const auto tokens = std::accumulate(seq_lengths.begin(), seq_lengths.end(), int32_t{0});
    const auto alternate_tokens =
        std::accumulate(alternate_seq_lengths.begin(), alternate_seq_lengths.end(), int32_t{0});
    const auto num_sequences = static_cast<int32_t>(seq_lengths.size());
    const auto alternate_num_sequences = static_cast<int32_t>(alternate_seq_lengths.size());
    const auto num_blocks = count_logical_blocks(seq_lengths, num_processed_tokens, cache_intervals);
    const auto alternate_num_blocks =
        count_logical_blocks(alternate_seq_lengths, alternate_processed_tokens, alternate_cache_intervals);

    const ov::Shape A_shape{static_cast<size_t>(num_heads)};
    const ov::Shape dt_shape{static_cast<size_t>(tokens), static_cast<size_t>(num_heads)};
    const ov::Shape alternate_dt_shape{static_cast<size_t>(alternate_tokens), static_cast<size_t>(num_heads)};
    const ov::Shape BC_shape{static_cast<size_t>(tokens),
                             static_cast<size_t>(num_groups),
                             static_cast<size_t>(state_size)};
    const ov::Shape alternate_BC_shape{static_cast<size_t>(alternate_tokens),
                                       static_cast<size_t>(num_groups),
                                       static_cast<size_t>(state_size)};
    const ov::Shape x_shape{static_cast<size_t>(tokens), static_cast<size_t>(num_heads), static_cast<size_t>(head_dim)};
    const ov::Shape alternate_x_shape{static_cast<size_t>(alternate_tokens),
                                      static_cast<size_t>(num_heads),
                                      static_cast<size_t>(head_dim)};
    const ov::Shape state_shape{static_cast<size_t>(num_blocks),
                                static_cast<size_t>(num_heads),
                                static_cast<size_t>(head_dim),
                                static_cast<size_t>(state_size)};
    const ov::Shape alternate_state_shape{static_cast<size_t>(alternate_num_blocks + 2),
                                          static_cast<size_t>(num_heads),
                                          static_cast<size_t>(head_dim),
                                          static_cast<size_t>(state_size)};
    const auto repeated = [](const ov::Shape& shape) {
        return std::vector<ov::Shape>{shape, shape, shape};
    };
    init_input_shapes(
        {InputShape{ov::PartialShape{num_heads}, repeated(A_shape)},
         InputShape{ov::PartialShape{-1, num_heads}, {dt_shape, alternate_dt_shape, dt_shape}},
         InputShape{ov::PartialShape{-1, num_groups, state_size}, {BC_shape, alternate_BC_shape, BC_shape}},
         InputShape{ov::PartialShape{-1, num_heads, head_dim}, {x_shape, alternate_x_shape, x_shape}},
         InputShape{ov::PartialShape{-1, num_groups, state_size}, {BC_shape, alternate_BC_shape, BC_shape}},
         InputShape{ov::PartialShape{-1, num_heads, head_dim, state_size},
                    {state_shape, alternate_state_shape, state_shape}},
         InputShape{ov::PartialShape::dynamic(1),
                    {ov::Shape{static_cast<size_t>(num_sequences + 1)},
                     ov::Shape{static_cast<size_t>(alternate_num_sequences + 1)},
                     ov::Shape{static_cast<size_t>(num_sequences + 1)}}},
         InputShape{ov::PartialShape::dynamic(1),
                    {ov::Shape{static_cast<size_t>(num_blocks)},
                     ov::Shape{static_cast<size_t>(alternate_num_blocks)},
                     ov::Shape{static_cast<size_t>(num_blocks)}}},
         InputShape{ov::PartialShape::dynamic(1),
                    {ov::Shape{static_cast<size_t>(num_sequences + 1)},
                     ov::Shape{static_cast<size_t>(alternate_num_sequences + 1)},
                     ov::Shape{static_cast<size_t>(num_sequences + 1)}}},
         InputShape{ov::PartialShape::dynamic(1),
                    {ov::Shape{static_cast<size_t>(num_sequences)},
                     ov::Shape{static_cast<size_t>(alternate_num_sequences)},
                     ov::Shape{static_cast<size_t>(num_sequences)}}},
         InputShape{ov::PartialShape::dynamic(1),
                    {ov::Shape{static_cast<size_t>(num_sequences)},
                     ov::Shape{static_cast<size_t>(alternate_num_sequences)},
                     ov::Shape{static_cast<size_t>(num_sequences)}}}});

    auto p_A = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{num_heads});
    auto p_dt = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads});
    auto p_B = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_groups, state_size});
    auto p_x = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads, head_dim});
    auto p_C = std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_groups, state_size});
    auto p_state =
        std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads, head_dim, state_size});
    auto p_subseq = std::make_shared<ov::op::v0::Parameter>(index_type, ov::PartialShape{-1});
    auto p_blocks = std::make_shared<ov::op::v0::Parameter>(index_type, ov::PartialShape{-1});
    auto p_block_begins = std::make_shared<ov::op::v0::Parameter>(index_type, ov::PartialShape{-1});
    auto p_num_processed = std::make_shared<ov::op::v0::Parameter>(index_type, ov::PartialShape{-1});
    auto p_cache_interval = std::make_shared<ov::op::v0::Parameter>(index_type, ov::PartialShape{-1});
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

    const auto& [num_heads,
                 num_groups,
                 head_dim,
                 state_size,
                 seq_lengths,
                 num_processed_tokens_param,
                 cache_intervals,
                 element_type,
                 index_type,
                 device] = GetParam();
    const auto num_sequences = static_cast<int32_t>(targetInputStaticShapes[9][0]);
    OPENVINO_ASSERT(targetInputStaticShapes[6][0] == static_cast<size_t>(num_sequences + 1));
    OPENVINO_ASSERT(targetInputStaticShapes[8][0] == static_cast<size_t>(num_sequences + 1));
    OPENVINO_ASSERT(targetInputStaticShapes[10][0] == static_cast<size_t>(num_sequences));
    const bool use_alternate_metadata = static_cast<size_t>(num_sequences) != seq_lengths.size();
    auto active_seq_lengths = seq_lengths;
    auto active_processed_tokens = num_processed_tokens_param;
    auto active_cache_intervals = cache_intervals;
    if (use_alternate_metadata) {
        active_seq_lengths.push_back(1);
        active_processed_tokens.push_back(0);
        active_cache_intervals.push_back(2);
    }
    OPENVINO_ASSERT(static_cast<size_t>(num_sequences) == active_seq_lengths.size());

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
        const int32_t seq_len = active_seq_lengths[seq];
        const int32_t seq_interval = active_cache_intervals[seq];
        const int32_t processed = active_processed_tokens[seq];

        subsequence_begins.push_back(subsequence_begins.back() + seq_len);
        num_processed_tokens.push_back(processed);
        cache_interval.push_back(seq_interval);

        int32_t required_slots = 1;
        if (seq_len == 0) {
            required_slots = 0;
        } else if (seq_interval > 0) {
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
            tensor = make_index_tensor(subsequence_begins, index_type);
        } else if (i == 7) {
            tensor = make_index_tensor(block_indices, index_type);
        } else if (i == 8) {
            tensor = make_index_tensor(block_indices_begins, index_type);
        } else if (i == 9) {
            tensor = make_index_tensor(num_processed_tokens, index_type);
        } else if (i == 10) {
            tensor = make_index_tensor(cache_interval, index_type);
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
    const auto& [num_heads,
                 num_groups,
                 head_dim,
                 state_size,
                 seq_lengths,
                 num_processed_tokens,
                 cache_intervals,
                 element_type,
                 index_type,
                 device] = GetParam();

    if (element_type == ov::element::f16) {
        return calculate_data_typed_refs<ov::float16>(
            host_inputs, function, num_heads, num_groups, head_dim, state_size, element_type, index_type);
    }
    if (element_type == ov::element::bf16) {
        return calculate_data_typed_refs<ov::bfloat16>(
            host_inputs, function, num_heads, num_groups, head_dim, state_size, element_type, index_type);
    }
    return calculate_data_typed_refs<float>(
        host_inputs, function, num_heads, num_groups, head_dim, state_size, element_type, index_type);
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
