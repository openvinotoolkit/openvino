// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "nodes/kernels/selective_ssm.hpp"

namespace ov::intel_cpu::node::kernel::test {

struct ReferenceResult {
    std::vector<float> output;
    std::vector<float> state;
};

inline std::vector<float> make_values(size_t count, float scale, float offset = 0.F) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i) {
        values[i] = offset + scale * static_cast<float>(static_cast<int>(i % 11) - 5);
    }
    return values;
}

template <typename T>
std::vector<T> cast_values(const std::vector<float>& values) {
    std::vector<T> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](float value) {
        return static_cast<T>(value);
    });
    return result;
}

template <typename T>
std::vector<float> to_float(const std::vector<T>& values) {
    std::vector<float> result(values.size());
    std::transform(values.begin(), values.end(), result.begin(), [](T value) {
        return static_cast<float>(value);
    });
    return result;
}

inline ReferenceResult reference_selective_ssm(const std::vector<float>& state_decay_rates,
                                               const std::vector<float>& time_steps,
                                               const std::vector<float>& input_projections,
                                               const std::vector<float>& input,
                                               const std::vector<float>& output_projections,
                                               const std::vector<float>& initial_state,
                                               const SelectiveSSMShape& shape) {
    ReferenceResult result;
    result.output.resize(shape.batch_size * shape.sequence_length * shape.num_heads * shape.head_dim);
    result.state = initial_state;
    const auto heads_per_group = shape.num_heads / shape.num_groups;
    const auto state_batch_stride = shape.num_heads * shape.head_dim * shape.state_size;
    const auto state_head_stride = shape.head_dim * shape.state_size;

    for (size_t batch = 0; batch < shape.batch_size; ++batch) {
        for (size_t token = 0; token < shape.sequence_length; ++token) {
            for (size_t head = 0; head < shape.num_heads; ++head) {
                const auto token_head = (batch * shape.sequence_length + token) * shape.num_heads + head;
                const auto group = head / heads_per_group;
                const auto projection_base =
                    ((batch * shape.sequence_length + token) * shape.num_groups + group) * shape.state_size;
                const auto state_base = batch * state_batch_stride + head * state_head_stride;
                const auto input_base = token_head * shape.head_dim;
                const float time_step = time_steps[token_head];
                const float decay = std::exp(state_decay_rates[head] * time_step);
                for (size_t position = 0; position < shape.head_dim; ++position) {
                    float value = 0.F;
                    for (size_t state_index = 0; state_index < shape.state_size; ++state_index) {
                        auto& state = result.state[state_base + position * shape.state_size + state_index];
                        state = state * decay + input[input_base + position] * time_step *
                                                    input_projections[projection_base + state_index];
                        value += state * output_projections[projection_base + state_index];
                    }
                    result.output[input_base + position] = value;
                }
            }
        }
    }
    return result;
}

inline CpuParallelPtr make_parallel() {
    return std::make_shared<CpuParallel>(TbbPartitioner::STATIC);
}

struct SelectiveSSMKernelTestArgs {
    const void* state_decay_rates = nullptr;
    const void* time_steps = nullptr;
    const void* input_projections = nullptr;
    const void* input = nullptr;
    const void* output_projections = nullptr;
    const void* initial_state = nullptr;
    void* output = nullptr;
    void* final_state = nullptr;
    SelectiveSSMShape shape{};
    element::Type data_precision = element::dynamic;
    float* state_scratch = nullptr;
    size_t head_dim_tile = 0;
    CpuParallelPtr cpu_parallel;
    const float* fp32_input_projections = nullptr;
    const float* fp32_output_projections = nullptr;
    bool use_fp32_projections = false;
};

struct PagedSelectiveSSMKernelTestArgs {
    const void* state_decay_rates = nullptr;
    const void* time_steps = nullptr;
    const void* input_projections = nullptr;
    const void* input = nullptr;
    const void* output_projections = nullptr;
    void* state_cache = nullptr;
    const void* subsequence_begins = nullptr;
    const void* block_indices = nullptr;
    const void* block_indices_begins = nullptr;
    const void* num_processed_tokens = nullptr;
    const void* cache_intervals = nullptr;
    void* output = nullptr;
    PagedSelectiveSSMShape shape{};
    element::Type data_precision = element::dynamic;
    element::Type index_precision = element::dynamic;
    float* state_scratch = nullptr;
    size_t head_dim_tile = 0;
    int32_t* metadata_validation_scratch = nullptr;
    CpuParallelPtr cpu_parallel;
    const float* fp32_input_projections = nullptr;
    const float* fp32_output_projections = nullptr;
};

using SelectiveSSMKernelRunner = std::function<void(const SelectiveSSMKernelTestArgs&)>;
using PagedSelectiveSSMKernelRunner = std::function<void(const PagedSelectiveSSMKernelTestArgs&)>;

void run_selective_ssm_differential_stress(const element::Type& precision,
                                           float tolerance,
                                           bool use_fp32_projections,
                                           const SelectiveSSMKernelRunner& run_kernel);

void run_paged_selective_ssm_differential_stress(const element::Type& precision,
                                                 const element::Type& index_precision,
                                                 float tolerance,
                                                 const PagedSelectiveSSMKernelRunner& run_kernel);

}  // namespace ov::intel_cpu::node::kernel::test
