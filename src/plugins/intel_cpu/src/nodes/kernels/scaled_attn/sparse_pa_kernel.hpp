// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/sparse_brgemm.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#ifdef OPENVINO_ARCH_X86_64
#    include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#endif

// Defined outside XARCH, like QuantizeParam, so the cross-compiled dispatcher can name it in the signature.
namespace ov::Extensions::Cpu {

/// @brief Layout of quantized (u8/u4) paged KV caches, matching PagedAttention (see executor_pa.cpp):
///  - by-token (key or value): each row is a sequence of groups `[scale f32][zp f32][group data]`, so the row
///    holds `S + 8 * groups` elements (u8) or `S + 16 * groups` (u4); `*_group_size == 0` means one group per row.
///  - by-channel (key only): each page is `f32 scale[S] | f32 zp[S] | data[block_size, S]`.
/// The caches are either both quantized (`enabled()`) or both plain f32/f16/bf16. The node writes the current-step
/// K/V into quantized caches before calling the kernel, so the kernel only reads them.
struct SparsePAQuant {
    ov::element::Type key_prec = ov::element::dynamic;  ///< u8/u4 if the key cache is quantized, else dynamic
    ov::element::Type value_prec = ov::element::dynamic;
    size_t key_group_size = 0;
    size_t value_group_size = 0;
    bool key_bychannel = false;
    size_t block_size = 0;  ///< tokens per page, without by-channel parameter rows; 0 means key_cache dim 2

    [[nodiscard]] bool enabled() const {
        return key_prec != ov::element::dynamic;
    }
};

/// @brief Byte layout of the per-thread scratch of the sparse PA kernel. The node allocates one such block per worker
/// thread (so nothing is allocated while computing) and the kernel carves it up with the same layout. Every part is
/// 64-byte aligned. Sizes are for the largest query-head pack, max_q_head_pack_size().
struct SparsePAScratchLayout {
    size_t q = 0;            ///< float[pack, k_head_size]: query rows of the pack, f32
    size_t scores = 0;       ///< float[pack, max_run_rows]: scores, then weights, of a run of key rows
    size_t raw = 0;          ///< float[pack, 4 * amx_unit_blocks * key_block_size]: AMX chunk scores / weights
    size_t runs = 0;         ///< 3 * int64[max_selected_blocks]: AMX run decomposition
    size_t k_run = 0;        ///< quantized caches: float[max_run_rows, k_head_size] dequantized key run
    size_t v_run = 0;        ///< quantized caches, pack > 1: float[max_run_rows, v_head_size] dequantized value run
    size_t k_dq = 0;         ///< quantized + AMX: data-type[4 * amx_unit_blocks * key_block_size, k_head_size]
    size_t v_dq = 0;         ///< quantized + AMX: data-type[4 * amx_unit_blocks * key_block_size, v_head_size]
    SparseBrgemmLayout brgemm;  ///< the tile buffers of the brgemm path
    size_t total = 0;           ///< bytes per thread

    /// @param max_run_rows most key rows of one run: the page size when adjacent selected blocks are merged inside a
    /// page, key_block_size when they are not
    /// @param amx_unit_blocks 0 when the AMX path is not used
    /// @param max_pack_size max_q_head_pack_size()
    /// @param data_size size in bytes of the query/key/value data type (the AMX scratch holds that type)
    SparsePAScratchLayout(size_t k_head_size,
                          size_t v_head_size,
                          size_t key_block_size,
                          size_t max_run_rows,
                          size_t max_selected_blocks,
                          size_t amx_unit_blocks,
                          size_t max_pack_size,
                          size_t data_size,
                          const SparsePAQuant& quant,
                          const SparseBrgemmSizes& brgemm_sizes = {}) {
        const size_t amx_rows = amx_unit_blocks * 4 * key_block_size;  // 0 when AMX is not used
        const bool amx_quant = amx_unit_blocks != 0 && quant.enabled();
        // Quantized keys are dequantized to f32 for by-channel keys and for packs; values for packs.
        const size_t k_run_rows = quant.enabled() && (quant.key_bychannel || max_pack_size > 1) ? max_run_rows : 0;
        const size_t v_run_rows = quant.enabled() && max_pack_size > 1 ? max_run_rows : 0;
        q = 0;
        scores = q + aligned(max_pack_size * k_head_size * sizeof(float));
        raw = scores + aligned(max_pack_size * max_run_rows * sizeof(float));
        runs = raw + aligned(max_pack_size * amx_rows * sizeof(float));
        k_run = runs + aligned(amx_unit_blocks != 0 ? 3 * max_selected_blocks * sizeof(int64_t) : 0);
        v_run = k_run + aligned(k_run_rows * k_head_size * sizeof(float));
        k_dq = v_run + aligned(v_run_rows * v_head_size * sizeof(float));
        v_dq = k_dq + aligned(amx_quant ? amx_rows * k_head_size * data_size : 0);
        brgemm = SparseBrgemmLayout(v_dq + aligned(amx_quant ? amx_rows * v_head_size * data_size : 0),
                                    k_head_size,
                                    v_head_size,
                                    max_pack_size,
                                    data_size,
                                    brgemm_sizes);
        total = brgemm.total;
    }
};

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

// Single cross-compiled entry point for ov::intel_cpu::node::SparsePA (see sparse_pa.h for the algorithm
// description). The node writes the current-step K/V into the caches first (paged_attn_memcpy /
// paged_attn_quantkv, parallel over token x kv head), so this reads them: it parallelizes the sparse attention over
// (global query token, query head), with one preallocated scratch block per worker thread (SparsePAScratchLayout).
// Query, output and plain caches share one floating-point type (f32/f16/bf16); quantized caches are described by
// `quant`. The physical page size is quant.block_size, or the key cache's dim 2 for plain caches.
void sparse_pa_compute(const ov::intel_cpu::PlainTensor& query,
                       const ov::intel_cpu::PlainTensor& key_cache,
                       const ov::intel_cpu::PlainTensor& value_cache,
                       const ov::intel_cpu::PlainTensor& block_indices,
                       const ov::intel_cpu::PlainTensor& block_indices_begins,
                       const ov::intel_cpu::PlainTensor& subsequence_begins,
                       const ov::intel_cpu::PlainTensor& past_lens,
                       const ov::intel_cpu::PlainTensor& sel_indices,
                       const ov::intel_cpu::PlainTensor& sel_count,
                       ov::intel_cpu::PlainTensor& out,
                       size_t q_head_num,
                       size_t k_head_num,
                       size_t k_head_size,
                       size_t v_head_size,
                       size_t key_block_size,
                       size_t sel_head_num,
                       bool causal,
                       float scale,
                       const SparsePAQuant& quant,
                       ov::element::Type amx_prec,
                       void* gemv_qk1,
                       void* gemv_qk4,
                       size_t amx_unit_blocks,
                       size_t max_run_rows,
                       const SparseBrgemm& brgemm,
                       uint8_t* scratch,
                       size_t scratch_bytes_per_thread,
                       int32_t* token_ids,
                       const ov::intel_cpu::CpuParallelPtr& cpu_parallel);

}  // namespace ov::Extensions::Cpu::XARCH