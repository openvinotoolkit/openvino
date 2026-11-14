// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "utils/plain_tensor.hpp"

// Defined outside XARCH so the cross-compiled dispatcher can name the types in the signature.
namespace ov::Extensions::Cpu {

/// @brief What is fixed for the lifetime of a SparsePA node. The caches are either both quantized (u8 / u4 in the
/// specs, with the PagedAttention layouts, see CacheSpec) or both plain f32 / f16 / bf16. Query and output share one
/// floating-point type; plain caches have it too, or are f16 next to f32 data (like PagedAttention).
struct SparsePAConfig {
    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t key_block_size = 0;
    bool causal = true;
    ov::element::Type data_precision = ov::element::f32;  ///< query and output type
    CacheSpec key_spec;
    CacheSpec value_spec;
    /// Tokens per physical page, without the parameter rows of by-channel key pages; 0: dim 2 of the key cache (plain
    /// caches).
    size_t page_size = 0;
    /// Most key rows of one run: the page size when adjacent selected blocks are merged inside a page, key_block_size
    /// when they are not.
    size_t max_run_rows = 0;
    /// The AMX prefill kernels; not enabled when the node is not eligible. They are owned by the node and must outlive
    /// the executor.
    TileBrgemm brgemm;
};

/// @brief The tensors of one call.
struct SparsePATensors {
    const ov::intel_cpu::PlainTensor& query;                 ///< [tokens, num_heads * k_head_size]
    const ov::intel_cpu::PlainTensor& key_cache;             ///< [num_blocks, num_kv_heads, page rows, row size]
    const ov::intel_cpu::PlainTensor& value_cache;           ///< the same for the values
    const ov::intel_cpu::PlainTensor& block_indices;         ///< page table
    const ov::intel_cpu::PlainTensor& block_indices_begins;  ///< [num_sequences + 1]
    const ov::intel_cpu::PlainTensor& subsequence_begins;    ///< [num_sequences + 1]
    const ov::intel_cpu::PlainTensor& past_lens;             ///< [num_sequences]
    const ov::intel_cpu::PlainTensor& sel_indices;           ///< [tokens, num_selection_heads, K_max]
    const ov::intel_cpu::PlainTensor& sel_count;             ///< [tokens, num_selection_heads]
    ov::intel_cpu::PlainTensor& out;                         ///< [tokens, num_heads * v_head_size]
    float scale = 1.0F;
};

/// @brief Sparse attention of the SparsePA node over the paged caches (see sparse_pa.h for the algorithm). The node
/// writes the current-step K/V into the caches first, so this only reads them. Built once by
/// the factory below; the node asks for the scratch size of a call, allocates it and passes it to execute.
class SparsePAExecutor {
public:
    virtual ~SparsePAExecutor() = default;
    [[nodiscard]] virtual size_t scratch_bytes(const SparsePATensors& tensors,
                                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
    virtual void execute(const SparsePATensors& tensors,
                         uint8_t* scratch,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
};

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

// Cross-compiled factory (AVX-512, AVX2 or plain C++ by the CPU).
std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> make_sparse_pa_executor(
    const ov::Extensions::Cpu::SparsePAConfig& config);

}  // namespace ov::Extensions::Cpu::XARCH
