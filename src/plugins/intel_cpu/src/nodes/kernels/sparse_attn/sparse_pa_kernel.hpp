// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "cache/multi_cache.h"
#include "cpu_parallel.hpp"
#include "nodes/kernels/scaled_attn/attn_brgemm_common.hpp"
#include "nodes/kernels/scaled_attn/cache_spec.hpp"
#include "onednn/iml_type_mapper.h"
#include "utils/plain_tensor.hpp"

// Defined outside XARCH so the cross-compiled dispatcher can name the types in the signature.
namespace ov::Extensions::Cpu {

struct SparsePAConfig {
    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t key_block_size = 0;
    bool causal = true;
    ov::element::Type data_precision = ov::element::f32;  ///< query and output type
    ov::intel_cpu::CacheSpec key_spec;
    ov::intel_cpu::CacheSpec value_spec;
    /// Tokens per physical page, without the parameter rows of by-channel key pages; 0: dim 2 of the key cache (plain
    /// caches).
    size_t page_size = 0;
    /// Most K/V rows one attend_kv_rows() call takes: the page size (a call stays inside one physical page, so adjacent
    /// selected blocks merge up to a page), key_block_size when the page size is unknown. It sizes the scores and the
    /// dequantized-row buffers.
    size_t max_attend_kv_rows = 0;
    /// The AMX prefill kernels; empty when the executor is not eligible. make_*_executor() fills it (the node leaves it
    /// empty). The config owns them, so an executor that copies the config keeps them alive.
    TileBrgemmKernels brgemm;
};

/// @brief The tensors of one call.
struct SparsePATensors {
    const ov::intel_cpu::PlainTensor& query;          ///< [tokens, num_heads * k_head_size]
    const ov::intel_cpu::PlainTensor& key;            ///< current-step keys [tokens, num_kv_heads * k_head_size]
    const ov::intel_cpu::PlainTensor& value;          ///< current-step values [tokens, num_kv_heads * v_head_size]
    const ov::intel_cpu::PlainTensor& key_cache;      ///< [num_blocks, num_kv_heads, page rows, row size]
    const ov::intel_cpu::PlainTensor& value_cache;    ///< the same for the values
    const ov::intel_cpu::PlainTensor& block_indices;  ///< page table
    const ov::intel_cpu::PlainTensor& block_indices_begins;  ///< [num_sequences + 1]
    const ov::intel_cpu::PlainTensor& subsequence_begins;    ///< [num_sequences + 1]
    const ov::intel_cpu::PlainTensor& past_lens;             ///< [num_sequences]
    const ov::intel_cpu::PlainTensor& sel_indices;           ///< [tokens, num_selection_heads, K_max]
    const ov::intel_cpu::PlainTensor& sel_count;             ///< [tokens, num_selection_heads]
    ov::intel_cpu::PlainTensor& out;                         ///< [tokens, num_heads * v_head_size]
    float scale = 1.0F;
};

class SparsePAExecutor {
public:
    virtual ~SparsePAExecutor() = default;
    /// The implementation type the node reports: the matrix extension of the prefill kernels when it has them, else
    /// the SIMD stages of the variant.
    [[nodiscard]] virtual ov::intel_cpu::impl_desc_type impl_type() const = 0;
    [[nodiscard]] virtual size_t scratch_bytes(const SparsePATensors& tensors,
                                               const ov::intel_cpu::CpuParallelPtr& cpu_parallel) const = 0;
    virtual void execute(const SparsePATensors& tensors,
                         const ov::intel_cpu::MemoryPtr& scratch,
                         const ov::intel_cpu::CpuParallelPtr& cpu_parallel) = 0;
};

}  // namespace ov::Extensions::Cpu

namespace ov::Extensions::Cpu::XARCH {

std::shared_ptr<ov::Extensions::Cpu::SparsePAExecutor> make_sparse_pa_executor(
    const ov::Extensions::Cpu::SparsePAConfig& config,
    const ov::intel_cpu::MultiCachePtr& cache);

}  // namespace ov::Extensions::Cpu::XARCH
