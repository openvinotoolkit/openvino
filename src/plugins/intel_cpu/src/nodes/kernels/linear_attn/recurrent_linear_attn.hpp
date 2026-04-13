// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "cpu_parallel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

void recurrent_linear_attn(const ov::intel_cpu::PlainTensor& query,
                           const ov::intel_cpu::PlainTensor& key,
                           const ov::intel_cpu::PlainTensor& value,
                           const ov::intel_cpu::PlainTensor& recurrent_state,
                           const ov::intel_cpu::PlainTensor& gate,
                           const ov::intel_cpu::PlainTensor& beta,
                           float q_l2_norm_eps,
                           float k_l2_norm_eps,
                           bool fuse_qk_l2norm,
                           ov::intel_cpu::PlainTensor& output_attn,
                           ov::intel_cpu::PlainTensor& output_recurrent_state,
                           float* temp_buffer,
                           const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

void recurrent_linear_attn_paged(const ov::intel_cpu::PlainTensor& query,
                                 const ov::intel_cpu::PlainTensor& key,
                                 const ov::intel_cpu::PlainTensor& value,
                                 ov::intel_cpu::PlainTensor& recurrent_state_table,
                                 const ov::intel_cpu::PlainTensor& gate,
                                 const ov::intel_cpu::PlainTensor& beta,
                                 const ov::intel_cpu::PlainTensor& subsequence_begins,
                                 const ov::intel_cpu::PlainTensor& block_indices,
                                 const ov::intel_cpu::PlainTensor& block_indices_begins,
                                 const ov::intel_cpu::PlainTensor& past_lens,
                                 const ov::intel_cpu::PlainTensor& cache_interval,
                                 float q_l2_norm_eps,
                                 float k_l2_norm_eps,
                                 bool fuse_qk_l2norm,
                                 ov::intel_cpu::PlainTensor& output_attn,
                                 float* temp_buffer,
                                 const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
