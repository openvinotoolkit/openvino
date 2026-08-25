// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "nodes/kernels/selective_ssm.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::kernel {

class JitKernelBase;

struct SelectiveSSMJitRuntimeArgs {
    const void* state_decay_rates = nullptr;
    const void* time_steps = nullptr;
    const float* input_projections = nullptr;
    const void* input = nullptr;
    const float* output_projections = nullptr;
    const void* initial_state = nullptr;
    void* output = nullptr;
    void* final_state = nullptr;
    node::kernel::SelectiveSSMShape shape{};
    ov::element::Type data_precision = ov::element::dynamic;
    float* state_scratch = nullptr;
    size_t head_dim_tile = 0;
    CpuParallelPtr cpu_parallel;
    const JitKernelBase* fp32_state_kernel = nullptr;
    const JitKernelBase* direct_state_kernel = nullptr;
};

struct PagedSelectiveSSMJitRuntimeArgs {
    const void* state_decay_rates = nullptr;
    const void* time_steps = nullptr;
    const float* input_projections = nullptr;
    const void* input = nullptr;
    const float* output_projections = nullptr;
    void* state_cache = nullptr;
    const void* subsequence_begins = nullptr;
    const void* block_indices = nullptr;
    const void* block_indices_begins = nullptr;
    const void* num_processed_tokens = nullptr;
    const void* cache_intervals = nullptr;
    void* output = nullptr;
    node::kernel::PagedSelectiveSSMShape shape{};
    ov::element::Type data_precision = ov::element::dynamic;
    ov::element::Type index_precision = ov::element::dynamic;
    float* state_scratch = nullptr;
    int32_t* metadata_validation_scratch = nullptr;
    size_t head_dim_tile = 0;
    CpuParallelPtr cpu_parallel;
    const JitKernelBase* fp32_state_kernel = nullptr;
    const JitKernelBase* direct_state_kernel = nullptr;
    const JitKernelBase* no_state_store_kernel = nullptr;
};

void selective_ssm_jit(const SelectiveSSMJitRuntimeArgs& args);

void paged_selective_ssm_jit(const PagedSelectiveSSMJitRuntimeArgs& args);

}  // namespace ov::intel_cpu::kernel
