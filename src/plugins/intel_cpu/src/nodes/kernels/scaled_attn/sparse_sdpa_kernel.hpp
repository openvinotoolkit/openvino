// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/sparse_brgemm.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#endif

namespace ov::Extensions::Cpu {

// Describes how the key (or value) cache rows are stored. A plain cache is read as is (any of f32/f16/bf16, see
// load_row_f32() in the kernel). A quantized cache (u8 / u4, per-token groups of `group_size` elements) holds the
// quantized rows in `cache` and the {scale, zero point} pairs of the groups in `scale_zp`, laid out as
// [L, B, H, 2 * D / group_size] -- the layout written by attn_quant_by_token() -- and the rows that are touched are
// dequantized on the fly: only the selected blocks and the causal tail, never the whole history.
// With `by_channel` (u8 only) the groups run along L instead: `group_size` consecutive tokens share one
// {scale, zero point} per channel, and `scale_zp` is [2 * ceil(L / group_size), B, H, D] -- the layout written by
// attn_quant_by_channel(): row 2 * g holds the scales of group g, row 2 * g + 1 its zero points.
struct SparseKVQuant {
    ov::element::Type prec = ov::element::dynamic;  // u8 / u4, anything else means "not quantized"
    size_t group_size = 0;                          // elements of a row per group, or tokens per group when by_channel
    bool by_channel = false;
    const ov::intel_cpu::PlainTensor* scale_zp = nullptr;

    [[nodiscard]] bool enabled() const {
        return scale_zp != nullptr && (prec == ov::element::u8 || prec == ov::element::u4);
    }
};

}  // namespace ov::Extensions::Cpu

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
                         const SparseBrgemm& brgemm,
                         const SparseBrgemmLayout& brgemm_layout,
                         uint8_t* brgemm_scratch,
                         size_t brgemm_bytes_per_thread,
                         const int32_t* key_length,
                         size_t key_length_batch_stride,
                         const SparseKVQuant& key_quant,
                         const SparseKVQuant& value_quant,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

// Dequantizes the rows [0, L) of the quantized cache (b, h) into dense f32 rows, `dst` is [L, D].
void sparse_kv_dequantize_rows(const ov::intel_cpu::PlainTensor& cache,
                               const SparseKVQuant& quant,
                               size_t b,
                               size_t h,
                               size_t L,
                               size_t D,
                               float* dst);

}  // namespace ov::Extensions::Cpu::XARCH
