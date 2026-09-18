// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <initializer_list>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node::kernel {

struct SelectiveSSMShape {
    size_t batch_size = 0;
    size_t sequence_length = 0;
    size_t num_heads = 0;
    size_t head_dim = 0;
    size_t num_groups = 0;
    size_t state_size = 0;
};

struct PagedSelectiveSSMShape {
    size_t token_count = 0;
    size_t num_heads = 0;
    size_t head_dim = 0;
    size_t num_groups = 0;
    size_t state_size = 0;
    size_t physical_block_count = 0;
    size_t logical_block_count = 0;
    size_t sequence_count = 0;
};

size_t checked_size_product(std::initializer_list<size_t> dimensions, const char* tensor_name);

size_t checked_size_sum(std::initializer_list<size_t> values, const char* buffer_name);

size_t get_scratch_head_dim(size_t head_dim, size_t state_size, size_t outer_work_items, size_t thread_count);

void selective_ssm(const void* A,
                   const void* dt,
                   const void* B,
                   const void* x,
                   const void* C,
                   const void* recurrent_state,
                   void* output,
                   void* output_recurrent_state,
                   const SelectiveSSMShape& shape,
                   const ov::element::Type& precision,
                   float* state_scratch,
                   size_t scratch_head_dim,
                   const CpuParallelPtr& cpu_parallel,
                   const float* converted_B = nullptr,
                   const float* converted_C = nullptr);

void paged_selective_ssm(const void* A,
                         const void* dt,
                         const void* B,
                         const void* x,
                         const void* C,
                         void* recurrent_state_table,
                         const void* subsequence_begins,
                         const void* block_indices,
                         const void* block_indices_begins,
                         const void* num_processed_tokens,
                         const void* cache_interval,
                         void* output,
                         const PagedSelectiveSSMShape& shape,
                         const ov::element::Type& data_precision,
                         const ov::element::Type& state_precision,
                         const ov::element::Type& index_precision,
                         float* state_scratch,
                         size_t scratch_head_dim,
                         int32_t* metadata_validation_scratch,
                         const CpuParallelPtr& cpu_parallel,
                         const float* converted_B = nullptr,
                         const float* converted_C = nullptr);

}  // namespace ov::intel_cpu::node::kernel
