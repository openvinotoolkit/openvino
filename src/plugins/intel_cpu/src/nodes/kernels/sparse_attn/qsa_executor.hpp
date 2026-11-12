// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// The interface between the QSAIndexer / PagedQSAIndexer nodes and their cross-compiled kernels. Nothing here is
// instruction-set specific: the nodes and the dispatchers include it, the kernels (qsa_indexer_kernel.cpp,
// paged_qsa_indexer_kernel.cpp, qsa_brgemm_kernel.cpp) implement it.
//
// A node builds an executor once (the factory picks the data type) and then, per call, fills a Call with the tensors of
// its ports, asks the executor for the scratch size, and runs it. Both ops are two stages:
//   prepare      appends the new keys to the caches and pools the blocks they complete
//   score        per token: scores the complete blocks it sees and keeps the block_topk best ones
// The executor scores the tokens itself; the brgemm scorer (x64, AVX-512) is a separate object that replaces that
// stage. The stages run one after the other and share one scratch, which is as large as the larger need of the two.

#include <cstddef>
#include <cstdint>

#include "cpu_parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::qsa {

class QsaBrgemm;  // x64 only, see qsa_brgemm.hpp

/// @brief What is fixed per node, for both ops.
struct Config {
    size_t key_block_size = 0;       ///< tokens that make a block
    size_t num_selection_heads = 0;  ///< selection heads of the output
    size_t block_topk = 0;           ///< blocks kept per token and selection head
    size_t head_dim = 0;             ///< channels of an index head
    size_t rotary_dim = 0;           ///< channels rotated by RoPE
    size_t num_threads = 0;          ///< worker threads of the node (sizes the scratch)
    float eps = 0.0F;                ///< RMSNorm epsilon
    float scale = 0.0F;              ///< multiplier of the scores
    ov::element::Type precision;     ///< of the floating-point ports (f32, f16 or bf16)
    /// The chosen blocks by descending score instead of ascending block id (dense op only).
    bool order_by_score = false;
};

/// @brief The tensors of a call of the QSAIndexer node (dense caches, see qsa_indexer_kernel.cpp).
struct DenseCall {
    ov::intel_cpu::PlainTensor q_idx;                                          ///< [B, L, H_iq, D]
    ov::intel_cpu::PlainTensor k_idx;                                          ///< [B, L, H_ik, D]
    ov::intel_cpu::PlainTensor rope_cos, rope_sin;                             ///< [1 | B, L, Dr]
    ov::intel_cpu::PlainTensor key_length;                                     ///< [1 | B]
    ov::intel_cpu::PlainTensor q_norm_weight, k_norm_weight;                   ///< [D]
    ov::intel_cpu::PlainTensor past_raw_k_in, past_rope_in, summary_cache_in;  ///< the state inputs (read only)
    ov::intel_cpu::PlainTensor summary_cache, past_raw_k, past_rope;           ///< the updated state
    ov::intel_cpu::PlainTensor sel_indices, sel_count;                         ///< [B, H_sel, L, K], [B, H_sel, L]
};

/// @brief The tensors of a call of the PagedQSAIndexer node (caches addressed through the PagedAttention page table).
struct PagedCall {
    ov::intel_cpu::PlainTensor q_idx;                                           ///< [tokens, H_iq * D]
    ov::intel_cpu::PlainTensor k_idx;                                           ///< [tokens, H_ik * D]
    ov::intel_cpu::PlainTensor rope_cos, rope_sin;                              ///< [tokens, Dr]
    ov::intel_cpu::PlainTensor q_norm_weight, k_norm_weight;                    ///< [D]
    ov::intel_cpu::PlainTensor past_lens, subsequence_begins;                   ///< [seqs], [seqs + 1]
    ov::intel_cpu::PlainTensor block_indices, block_indices_begins;             ///< page table
    ov::intel_cpu::PlainTensor raw_k_cache, indexer_rope_cache, summary_cache;  ///< paged caches, written in place
    ov::intel_cpu::PlainTensor sel_indices, sel_count;                          ///< [tokens, H_sel, K], [tokens, H_sel]
};

/// @brief Appends the new keys of a call to the caches and pools the blocks they complete; optionally scores the
/// tokens.
template <typename Call>
class Executor {
public:
    virtual ~Executor() = default;
    /// Scratch bytes of a call (for both stages); also checks that the call fits its caches.
    [[nodiscard]] virtual size_t scratch_bytes(const Call& call) const = 0;
    virtual void prepare(const Call& call, uint8_t* scratch, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) = 0;
    virtual void score_tokens(const Call& call,
                              uint8_t* scratch,
                              const ov::intel_cpu::CpuParallelPtr& cpu_parallel) = 0;
};

/// @brief The scoring stage on brgemm (AVX-512 only): runs after Executor::prepare() on the same scratch.
template <typename Call>
class Scorer {
public:
    virtual ~Scorer() = default;
    /// Whether the call has anything to score (otherwise the executor's own scoring runs).
    [[nodiscard]] virtual bool applies(const Call& call) const = 0;
    [[nodiscard]] virtual size_t scratch_bytes(const Call& call) const = 0;
    virtual void score(const Call& call, uint8_t* scratch, const ov::intel_cpu::CpuParallelPtr& cpu_parallel) = 0;
};

using DenseExecutor = Executor<DenseCall>;
using PagedExecutor = Executor<PagedCall>;
using DenseScorer = Scorer<DenseCall>;
using PagedScorer = Scorer<PagedCall>;

}  // namespace ov::Extensions::Cpu::qsa
