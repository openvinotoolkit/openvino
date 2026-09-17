// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/paged_selective_ssm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <sstream>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "utils/precision_support.h"

namespace {
enum class InputPort : size_t {
    A,
    TimeStep,
    InputProjection,
    Input,
    OutputProjection,
    State,
    SubsequenceBegins,
    BlockIndices,
    BlockIndicesBegins,
    NumProcessedTokens,
    CacheInterval,
    Count,
};

constexpr size_t input_port_index(InputPort port) noexcept {
    return static_cast<size_t>(port);
}

constexpr bool is_float_port(InputPort port) noexcept {
    return port <= InputPort::State;
}

inline constexpr size_t input_count = input_port_index(InputPort::Count);

template <typename DataT, typename StateT, typename IndexT>
void run_reference(const std::vector<DataT>& A,
                   const std::vector<DataT>& dt,
                   const std::vector<DataT>& B,
                   const std::vector<DataT>& x,
                   const std::vector<DataT>& C,
                   std::vector<StateT>& recurrent_state_table,
                   const std::vector<IndexT>& subsequence_begins,
                   const std::vector<IndexT>& block_indices,
                   const std::vector<IndexT>& block_indices_begins,
                   const std::vector<IndexT>& num_processed_tokens,
                   const std::vector<IndexT>& cache_interval,
                   size_t num_heads,
                   size_t num_groups,
                   size_t head_dim,
                   size_t state_size,
                   std::vector<DataT>& output) {
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
                    input_projection[state_index] = delta * static_cast<float>(B[projection_base + state_index]);
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
                    output[(token * num_heads + head) * head_dim + position] = static_cast<DataT>(reduction);
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
                        const auto write_block = static_cast<size_t>(block_indices[logical_block_begin + write_slot]);
                        for (size_t position = 0; position < head_dim; ++position) {
                            for (size_t state_index = 0; state_index < state_size; ++state_index) {
                                recurrent_state_table[state_off(write_block, head, position, state_index)] =
                                    static_cast<StateT>(state[position * state_size + state_index]);
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

template <typename T>
ov::Tensor make_index_tensor(const std::vector<T>& values, const ov::element::Type& index_type) {
    ov::Tensor tensor(index_type, ov::Shape{values.size()});
    if (index_type == ov::element::i32) {
        std::transform(values.begin(), values.end(), tensor.data<int32_t>(), [](T value) {
            return static_cast<int32_t>(value);
        });
    } else {
        std::transform(values.begin(), values.end(), tensor.data<int64_t>(), [](T value) {
            return static_cast<int64_t>(value);
        });
    }
    return tensor;
}

ov::Tensor make_f32_tensor(const ov::Shape& shape, const std::vector<float>& values) {
    ov::Tensor tensor(ov::element::f32, shape);
    OPENVINO_ASSERT(tensor.get_size() == values.size());
    std::copy(values.begin(), values.end(), tensor.data<float>());
    return tensor;
}

std::shared_ptr<ov::Model> make_paged_validation_model(size_t token_count = 1,
                                                       size_t physical_block_count = 2,
                                                       const ov::element::Type& index_type = ov::element::i32) {
    auto A = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto dt = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{token_count, 1});
    auto B = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{token_count, 1, 1});
    auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{token_count, 1, 1});
    auto C = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{token_count, 1, 1});
    auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{physical_block_count, 1, 1, 1});
    auto subsequences = std::make_shared<ov::op::v0::Parameter>(index_type, ov::Shape{2});
    auto blocks = std::make_shared<ov::op::v0::Parameter>(index_type, ov::Shape{physical_block_count});
    auto block_begins = std::make_shared<ov::op::v0::Parameter>(index_type, ov::Shape{2});
    auto processed = std::make_shared<ov::op::v0::Parameter>(index_type, ov::Shape{1});
    auto intervals = std::make_shared<ov::op::v0::Parameter>(index_type, ov::Shape{1});
    const ov::ParameterVector
        parameters{A, dt, B, x, C, state, subsequences, blocks, block_begins, processed, intervals};
    const auto ssm = std::make_shared<ov::op::internal::PagedSelectiveSSM>(A,
                                                                           dt,
                                                                           B,
                                                                           x,
                                                                           C,
                                                                           state,
                                                                           subsequences,
                                                                           blocks,
                                                                           block_begins,
                                                                           processed,
                                                                           intervals);
    return std::make_shared<ov::Model>(ssm->outputs(), parameters);
}

template <typename DataT, typename StateT, typename IndexT>
std::vector<ov::Tensor> calculate_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                             const std::shared_ptr<ov::Model>& function,
                                             size_t num_heads,
                                             size_t num_groups,
                                             size_t head_dim,
                                             size_t state_size,
                                             const ov::element::Type& data_type,
                                             const ov::element::Type& state_type) {
    const auto& params = function->get_parameters();
    const auto tensor_for = [&](InputPort port) -> const ov::Tensor& {
        return host_inputs.at(params.at(input_port_index(port)));
    };

    auto A = tensor_to_vector<DataT>(tensor_for(InputPort::A));
    auto dt = tensor_to_vector<DataT>(tensor_for(InputPort::TimeStep));
    auto B = tensor_to_vector<DataT>(tensor_for(InputPort::InputProjection));
    auto x = tensor_to_vector<DataT>(tensor_for(InputPort::Input));
    auto C = tensor_to_vector<DataT>(tensor_for(InputPort::OutputProjection));
    auto state = tensor_to_vector<StateT>(tensor_for(InputPort::State));
    auto subsequence_begins = tensor_to_vector<IndexT>(tensor_for(InputPort::SubsequenceBegins));
    auto block_indices = tensor_to_vector<IndexT>(tensor_for(InputPort::BlockIndices));
    auto block_indices_begins = tensor_to_vector<IndexT>(tensor_for(InputPort::BlockIndicesBegins));
    auto num_processed_tokens = tensor_to_vector<IndexT>(tensor_for(InputPort::NumProcessedTokens));
    auto cache_interval = tensor_to_vector<IndexT>(tensor_for(InputPort::CacheInterval));

    std::vector<DataT> ref_output;
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

    ov::Tensor output_tensor(data_type, tensor_for(InputPort::Input).get_shape());
    std::copy(ref_output.begin(), ref_output.end(), output_tensor.data<DataT>());

    ov::Tensor state_tensor(state_type, tensor_for(InputPort::State).get_shape());
    std::copy(state.begin(), state.end(), state_tensor.data<StateT>());

    return {output_tensor, state_tensor};
}

template <typename DataT, typename StateT>
std::vector<ov::Tensor> calculate_state_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                                   const std::shared_ptr<ov::Model>& function,
                                                   size_t num_heads,
                                                   size_t num_groups,
                                                   size_t head_dim,
                                                   size_t state_size,
                                                   const ov::element::Type& data_type,
                                                   const ov::element::Type& state_type,
                                                   const ov::element::Type& index_type) {
    if (index_type == ov::element::i64) {
        return calculate_typed_refs<DataT, StateT, int64_t>(host_inputs,
                                                            function,
                                                            num_heads,
                                                            num_groups,
                                                            head_dim,
                                                            state_size,
                                                            data_type,
                                                            state_type);
    }
    return calculate_typed_refs<DataT, StateT, int32_t>(host_inputs,
                                                        function,
                                                        num_heads,
                                                        num_groups,
                                                        head_dim,
                                                        state_size,
                                                        data_type,
                                                        state_type);
}

template <typename DataT>
std::vector<ov::Tensor> calculate_data_typed_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& host_inputs,
                                                  const std::shared_ptr<ov::Model>& function,
                                                  size_t num_heads,
                                                  size_t num_groups,
                                                  size_t head_dim,
                                                  size_t state_size,
                                                  const ov::element::Type& data_type,
                                                  const ov::element::Type& state_type,
                                                  const ov::element::Type& index_type) {
    if (state_type == ov::element::f32) {
        return calculate_state_typed_refs<DataT, float>(host_inputs,
                                                        function,
                                                        num_heads,
                                                        num_groups,
                                                        head_dim,
                                                        state_size,
                                                        data_type,
                                                        state_type,
                                                        index_type);
    }
    if (state_type == ov::element::f16) {
        return calculate_state_typed_refs<DataT, ov::float16>(host_inputs,
                                                              function,
                                                              num_heads,
                                                              num_groups,
                                                              head_dim,
                                                              state_size,
                                                              data_type,
                                                              state_type,
                                                              index_type);
    }
    if (state_type == ov::element::bf16) {
        return calculate_state_typed_refs<DataT, ov::bfloat16>(host_inputs,
                                                               function,
                                                               num_heads,
                                                               num_groups,
                                                               head_dim,
                                                               state_size,
                                                               data_type,
                                                               state_type,
                                                               index_type);
    }
    OPENVINO_THROW("Unsupported PagedSelectiveSSM state precision ", state_type, ".");
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
                 state_type,
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
    result << "_StateType=" << state_type;
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
                 state_type,
                 index_type,
                 device] = GetParam();

    targetDevice = device;
    this->data_type = data_type;
    this->state_type = state_type;
    if (!ov::intel_cpu::hasHardwareSupport(state_type)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << state_type << " state storage on this system.";
    }
    configuration[ov::hint::inference_precision.name()] = state_type;
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
    const ov::Shape state_shape{static_cast<size_t>(num_blocks) + 2,
                                static_cast<size_t>(num_heads),
                                static_cast<size_t>(head_dim),
                                static_cast<size_t>(state_size)};
    const ov::Shape alternate_state_shape{static_cast<size_t>(alternate_num_blocks) + 2,
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
    // Keep the source model valid. ConvertPagedAttnInputs applies the requested state storage precision inside the
    // CPU pipeline while computation parameters retain data_type through keep_const_precision.
    auto p_state =
        std::make_shared<ov::op::v0::Parameter>(data_type, ov::PartialShape{-1, num_heads, head_dim, state_size});
    for (const auto& parameter : {p_A, p_dt, p_B, p_x, p_C, p_state}) {
        ov::enable_keep_const_precision(parameter);
    }
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
                 state_type,
                 index_type,
                 device] = GetParam();
    const auto shape_for = [&](InputPort port) -> const ov::Shape& {
        return targetInputStaticShapes.at(input_port_index(port));
    };
    OPENVINO_ASSERT(targetInputStaticShapes.size() == input_count);
    const auto num_sequences = static_cast<int32_t>(shape_for(InputPort::NumProcessedTokens).front());
    OPENVINO_ASSERT(shape_for(InputPort::SubsequenceBegins).front() == static_cast<size_t>(num_sequences + 1));
    OPENVINO_ASSERT(shape_for(InputPort::BlockIndicesBegins).front() == static_cast<size_t>(num_sequences + 1));
    OPENVINO_ASSERT(shape_for(InputPort::CacheInterval).front() == static_cast<size_t>(num_sequences));
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

    int32_t next_physical_block = 0;
    for (int32_t seq = 0; seq < num_sequences; seq++) {
        const int32_t seq_len = active_seq_lengths[seq];
        const int32_t seq_interval = active_cache_intervals[seq];
        const int32_t processed = active_processed_tokens[seq];

        subsequence_begins.push_back(subsequence_begins.back() + seq_len);
        num_processed_tokens.push_back(processed);
        cache_interval.push_back(seq_interval);

        if (seq_len > 0) {
            const int32_t read_block = next_physical_block++;
            block_indices.push_back(read_block);
        }
        if (seq_len > 0 && seq_interval > 0) {
            const int32_t prev_nums = processed % seq_interval;
            const int32_t write_blocks = (prev_nums + seq_len + seq_interval - 1) / seq_interval;
            const bool alias_first_write = processed == 0 || prev_nums != 0;
            const int32_t read_block = block_indices.back();
            for (int32_t write = 0; write < write_blocks; ++write) {
                block_indices.push_back(write == 0 && alias_first_write ? read_block : next_physical_block++);
            }
        }
        block_indices_begins.push_back(static_cast<int32_t>(block_indices.size()));
    }

    const auto& params = function->get_parameters();
    const bool use_remote_tensors = targetDevice == "GPU";
    ov::RemoteContext remote_context;
    if (use_remote_tensors) {
        remote_context = compiledModel.get_context();
    }

    for (size_t input_index = 0; input_index < params.size(); ++input_index) {
        const auto port = static_cast<InputPort>(input_index);
        const auto& param = params[input_index];
        const auto& shape = targetInputStaticShapes[input_index];
        ov::Tensor tensor;

        if (port == InputPort::A) {
            tensor = ov::test::utils::create_and_fill_tensor_real_distribution(param->get_element_type(),
                                                                               shape,
                                                                               -0.5f,
                                                                               0.2f,
                                                                               1);
        } else if (port == InputPort::TimeStep) {
            tensor = ov::test::utils::create_and_fill_tensor_real_distribution(param->get_element_type(),
                                                                               shape,
                                                                               0.0f,
                                                                               0.5f,
                                                                               1);
        } else if (is_float_port(port)) {
            const auto tensor_type = port == InputPort::State ? state_type : param->get_element_type();
            tensor = ov::test::utils::create_and_fill_tensor(tensor_type,
                                                             shape,
                                                             ov::test::utils::InputGenerateData(-0.5f, 1.0f, 1000, 1));
        } else if (port == InputPort::SubsequenceBegins) {
            tensor = make_index_tensor(subsequence_begins, index_type);
        } else if (port == InputPort::BlockIndices) {
            tensor = make_index_tensor(block_indices, index_type);
        } else if (port == InputPort::BlockIndicesBegins) {
            tensor = make_index_tensor(block_indices_begins, index_type);
        } else if (port == InputPort::NumProcessedTokens) {
            tensor = make_index_tensor(num_processed_tokens, index_type);
        } else if (port == InputPort::CacheInterval) {
            tensor = make_index_tensor(cache_interval, index_type);
        } else {
            OPENVINO_THROW("Unexpected PagedSelectiveSSM input port ", input_index, ".");
        }

        // The operation updates the state input in place, so the oracle needs an independent pre-inference snapshot.
        ov::Tensor host_tensor(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(host_tensor);
        host_inputs[param] = host_tensor;
        if (use_remote_tensors && is_float_port(port)) {
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
                 state_type,
                 index_type,
                 device] = GetParam();

    if (element_type == ov::element::f16) {
        return calculate_data_typed_refs<ov::float16>(host_inputs,
                                                      function,
                                                      num_heads,
                                                      num_groups,
                                                      head_dim,
                                                      state_size,
                                                      element_type,
                                                      state_type,
                                                      index_type);
    }
    if (element_type == ov::element::bf16) {
        return calculate_data_typed_refs<ov::bfloat16>(host_inputs,
                                                       function,
                                                       num_heads,
                                                       num_groups,
                                                       head_dim,
                                                       state_size,
                                                       element_type,
                                                       state_type,
                                                       index_type);
    }
    return calculate_data_typed_refs<float>(host_inputs,
                                            function,
                                            num_heads,
                                            num_groups,
                                            head_dim,
                                            state_size,
                                            element_type,
                                            state_type,
                                            index_type);
}

std::vector<ov::Tensor> PagedSelectiveSSMLayerTest::get_plugin_outputs() {
    auto outputs = SubgraphBaseTest::get_plugin_outputs();

    const auto& state_param = function->get_parameters().at(input_port_index(InputPort::State));
    const auto actual_state_tensor = inferRequest.get_tensor(state_param);
    ov::Tensor host_state_tensor(actual_state_tensor.get_element_type(), actual_state_tensor.get_shape());
    actual_state_tensor.copy_to(host_state_tensor);
    outputs.push_back(host_state_tensor);

    return outputs;
}

void PagedSelectiveSSMLayerTest::compare(const std::vector<ov::Tensor>& expected,
                                         const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    if (data_type == ov::element::bf16 || state_type == ov::element::bf16) {
        abs_threshold = 1e-3f;
        rel_threshold = 1e-2f;
    } else if (data_type == ov::element::f16 || state_type == ov::element::f16) {
        abs_threshold = 5e-4f;
        rel_threshold = 1e-5f;
    } else {
        abs_threshold = 2e-4f;
        rel_threshold = 1e-5f;
    }
    ov::test::utils::compare(expected[0], actual[0], abs_threshold, rel_threshold);
    ov::test::utils::compare(expected[1], actual[1], abs_threshold, rel_threshold);
}

TEST_P(PagedSelectiveSSMLayerTest, Inference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    const auto runtime_model = compiledModel.get_runtime_model();
    CheckNumberOfNodesWithType(runtime_model, {"PagedSelectiveSSM"}, 1);
    CheckNumberOfNodesWithType(runtime_model, {"Loop"}, 0);
}

TEST(PagedSelectiveSSMFunctionalTest, AlignsComputationPrecisionAfterCpuLowering) {
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::bf16)) {
        GTEST_SKIP() << "CPU does not support bf16 inference.";
    }

    auto model = make_paged_validation_model();
    const auto a_parameter = model->get_parameters().at(input_port_index(InputPort::A));
    const auto a_constant = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {-0.2F});
    a_parameter->output(0).replace(a_constant->output(0));
    model->remove_parameter(a_parameter);
    const auto result = model->get_results().front();
    const auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1.F});
    const auto matmul = std::make_shared<ov::op::v0::MatMul>(result->input_value(0), weights, false, false);
    result->input(0).replace_source_output(matmul);

    ov::Core core;
    const auto compiled_model = core.compile_model(model, "CPU", ov::hint::inference_precision(ov::element::bf16));
    const auto runtime_model = compiled_model.get_runtime_model();
    CheckNumberOfNodesWithType(runtime_model, {"PagedSelectiveSSM"}, 1);
    for (const auto& node : runtime_model->get_ops()) {
        const auto& rt_info = node->get_rt_info();
        if (rt_info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == "PagedSelectiveSSM") {
            EXPECT_EQ(rt_info.at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>(), ov::element::bf16);
        }
    }
}

TEST(PagedSelectiveSSMFunctionalTest, RejectsMalformedMetadataBeforeExecution) {
    struct MetadataCase {
        const char* name;
        std::vector<int32_t> subsequences;
        std::vector<int32_t> blocks;
        std::vector<int32_t> block_begins;
        std::vector<int32_t> processed;
        std::vector<int32_t> intervals;
    };

    const std::vector<MetadataCase> cases{
        {"negative sequence offset", {-1, 1}, {0, 1}, {0, 2}, {0}, {1}},
        {"wrong final sequence offset", {0, 0}, {0, 1}, {0, 2}, {0}, {1}},
        {"negative block index", {0, 1}, {-1, 1}, {0, 2}, {0}, {1}},
        {"out of range block index", {0, 1}, {0, 2}, {0, 2}, {0}, {1}},
        {"negative processed token count", {0, 1}, {0, 1}, {0, 2}, {-1}, {1}},
        {"insufficient writable blocks", {0, 1}, {0, 1}, {0, 1}, {0}, {1}},
    };

    ov::Core core;
    auto compiled_model =
        core.compile_model(make_paged_validation_model(), "CPU", ov::hint::inference_precision(ov::element::f32));
    for (const auto& test_case : cases) {
        SCOPED_TRACE(test_case.name);
        auto request = compiled_model.create_infer_request();
        const auto set_input = [&](InputPort port, const ov::Tensor& tensor) {
            request.set_input_tensor(input_port_index(port), tensor);
        };
        set_input(InputPort::A, make_f32_tensor({1}, {-0.2F}));
        set_input(InputPort::TimeStep, make_f32_tensor({1, 1}, {0.1F}));
        set_input(InputPort::InputProjection, make_f32_tensor({1, 1, 1}, {0.2F}));
        set_input(InputPort::Input, make_f32_tensor({1, 1, 1}, {0.3F}));
        set_input(InputPort::OutputProjection, make_f32_tensor({1, 1, 1}, {0.4F}));
        set_input(InputPort::State, make_f32_tensor({2, 1, 1, 1}, {0.F, 0.F}));
        set_input(InputPort::SubsequenceBegins, make_index_tensor(test_case.subsequences, ov::element::i32));
        set_input(InputPort::BlockIndices, make_index_tensor(test_case.blocks, ov::element::i32));
        set_input(InputPort::BlockIndicesBegins, make_index_tensor(test_case.block_begins, ov::element::i32));
        set_input(InputPort::NumProcessedTokens, make_index_tensor(test_case.processed, ov::element::i32));
        set_input(InputPort::CacheInterval, make_index_tensor(test_case.intervals, ov::element::i32));
        EXPECT_THROW(request.infer(), ov::Exception);
    }
}

}  // namespace ov::test
