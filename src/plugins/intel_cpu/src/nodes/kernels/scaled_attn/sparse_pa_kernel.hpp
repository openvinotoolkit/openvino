// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

// Single cross-compiled entry point for ov::intel_cpu::node::SparsePA (see sparse_pa.h for the
// algorithm description). The node's execute() does nothing but marshal MemoryPtrs into
// PlainTensors and call this once -- mirroring nodes/paged_gated_delta_net.cpp's
// recurrent_linear_attn_paged() entry point. Every other compute step (Q0 in-place KV-cache write,
// scalar online-softmax reference path, AMX-accelerated QK^T + vectorized P@V fast path, row
// precision conversion) lives entirely inside sparse_pa_kernel.cpp, so it is genuinely vectorized
// by cross_compiled_file()'s per-ISA clones instead of silently falling back to scalar code the way
// calling header-only HAVE_AVX512F/HAVE_AVX2-guarded templates directly from a plain translation
// unit would. key_cache/value_cache are mutated in place (Q0), so they are non-const refs; all
// other tensors are read-only for this call. The physical page size is key_cache's dim 2.
void sparse_pa_compute(const ov::intel_cpu::PlainTensor& query,
                       const ov::intel_cpu::PlainTensor& key,
                       const ov::intel_cpu::PlainTensor& value,
                       ov::intel_cpu::PlainTensor& key_cache,
                       ov::intel_cpu::PlainTensor& value_cache,
                       const ov::intel_cpu::PlainTensor& block_indices,
                       const ov::intel_cpu::PlainTensor& block_indices_begins,
                       const ov::intel_cpu::PlainTensor& subsequence_begins,
                       const ov::intel_cpu::PlainTensor& past_lens,
                       const ov::intel_cpu::PlainTensor& sel_indices,
                       const ov::intel_cpu::PlainTensor& sel_count,
                       const ov::intel_cpu::PlainTensor& qb_start,
                       ov::intel_cpu::PlainTensor& out,
                       size_t num_heads,
                       size_t num_kv_heads,
                       size_t k_head_size,
                       size_t v_head_size,
                       size_t key_block_size,
                       size_t num_selection_heads,
                       bool causal,
                       float scale,
                       ov::element::Type amx_prec,
                       void* gemv_qk1,
                       void* gemv_qk4,
                       size_t amx_unit_blocks,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH