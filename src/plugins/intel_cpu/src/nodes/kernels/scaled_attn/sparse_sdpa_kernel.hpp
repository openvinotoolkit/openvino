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

// Single cross-compiled entry point for ov::intel_cpu::node::SparseSDPA (see sparse_sdpa.h for the
// algorithm description). The node's execute() does nothing but marshal MemoryPtrs into
// PlainTensors, resolve its own GraphContext-owned DnnlScratchPad allocation (scratch sizing/
// allocation needs GraphContext, which kernels/ must never depend on), and call this once --
// mirroring nodes/paged_gated_delta_net.cpp's recurrent_linear_attn_paged() entry point. Every
// other compute step (scalar online-softmax reference path, AMX-accelerated QK^T + vectorized P@V
// fast path, row precision conversion) lives entirely inside sparse_sdpa_kernel.cpp, so it is
// genuinely vectorized by cross_compiled_file()'s per-ISA clones instead of silently falling back
// to scalar code the way calling header-only HAVE_AVX512F/HAVE_AVX2-guarded templates directly
// from a plain translation unit would.
void sparse_sdpa_compute(const ov::intel_cpu::PlainTensor& query,
                         const ov::intel_cpu::PlainTensor& key,
                         const ov::intel_cpu::PlainTensor& value,
                         const ov::intel_cpu::PlainTensor& sel_indices,
                         const ov::intel_cpu::PlainTensor& sel_count,
                         ov::intel_cpu::PlainTensor& out,
                         size_t num_heads,
                         size_t num_kv_heads,
                         size_t k_head_size,
                         size_t v_head_size,
                         size_t sel_block_size,
                         size_t h_sel,
                         bool causal,
                         float scale,
                         float* scratch_floats,
                         size_t per_thread_floats,
                         ov::element::Type amx_prec,
                         void* gemv_qk1,
                         void* gemv_qk4,
                         size_t amx_unit_blocks,
                         int64_t* scratch_runs,
                         size_t runs_per_thread,
                         const int32_t* key_length,
                         size_t key_length_batch_stride,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH
