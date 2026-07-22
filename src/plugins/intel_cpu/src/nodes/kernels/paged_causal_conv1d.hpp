// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::Extensions::Cpu::XARCH {

// Public entry. The CPU plugin registers this function with `cross_compiled_file`, so a
// per-ISA (AVX512F/AVX2/ANY) copy is built and the runtime dispatcher picks the right one.
// Inside, runtime precision dispatch selects the (data, state) template instantiation.
void paged_causal_conv1d_exec(const void* input_embeds,
                              void* conv_state_table,
                              const void* conv_weight,
                              const void* conv_bias,
                              bool has_bias,
                              const int32_t* subsequence_begins,
                              const int32_t* block_indices,
                              const int32_t* block_indices_begins,
                              const int32_t* past_lens,
                              const int32_t* cache_interval,
                              void* output_embeds,
                              size_t batch_size_in_tokens,
                              size_t hidden_size,
                              size_t kernel_size,
                              size_t seq_count,
                              ov::element::Type data_precision,
                              ov::element::Type state_precision,
                              float* local_state,
                              const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
