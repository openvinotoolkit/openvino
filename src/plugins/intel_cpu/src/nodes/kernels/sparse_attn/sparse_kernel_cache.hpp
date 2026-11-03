// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Kernel caches of the sparse attention nodes (SparseSDPA, SparsePA): the JIT / brgemm kernels depend only on the
// shapes and the data type below, not on the node, so nodes with the same parameters (e.g. the attention layers of a
// model) share them through the params cache of the graph context, like the brgemm kernels of
// ScaledDotProductAttention.

#ifdef OPENVINO_ARCH_X86_64

#    include <common/utils.hpp>
#    include <cstddef>
#    include <memory>

#    include "cache/multi_cache.h"
#    include "nodes/kernels/scaled_attn/executor_pa_common.hpp"
#    include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#    include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

/// @brief Parameters of the brgemm kernels of the prefill path (see SparseBrgemmKernels).
struct SparseBrgemmKey {
    size_t pack = 0;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t tile_rows = 0;
    ov::element::Type prec;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        size_t seed = 0;
        seed = hash_combine(seed, pack);
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, tile_rows);
        seed = hash_combine(seed, prec.hash());
        return seed;
    }
    bool operator==(const SparseBrgemmKey& rhs) const {
        return pack == rhs.pack && k_head_size == rhs.k_head_size && v_head_size == rhs.v_head_size &&
               tile_rows == rhs.tile_rows && prec == rhs.prec;
    }
};

/// @brief Parameters of the matrix-vector AMX kernel of the QK^T of the legacy path (see JitMatMulVecAMX).
struct SparseGemvKey {
    int head_size = 0;
    int block_size = 0;
    ov::element::Type prec;

    [[nodiscard]] size_t hash() const {
        using namespace dnnl::impl;
        size_t seed = 0;
        seed = hash_combine(seed, head_size);
        seed = hash_combine(seed, block_size);
        seed = hash_combine(seed, prec.hash());
        return seed;
    }
    bool operator==(const SparseGemvKey& rhs) const {
        return head_size == rhs.head_size && block_size == rhs.block_size && prec == rhs.prec;
    }
};

/// @brief The brgemm kernels of the prefill path for the pack of query heads, created once per set of parameters. The
/// result is a copy of the cached bundle: it shares the kernels, and its handle points at them.
inline ov::Extensions::Cpu::SparseBrgemmKernels get_sparse_brgemm_kernels(const MultiCachePtr& cache,
                                                                          size_t pack,
                                                                          size_t k_head_size,
                                                                          size_t v_head_size,
                                                                          size_t tile_rows,
                                                                          ov::element::Type prec) {
    const SparseBrgemmKey key{pack, k_head_size, v_head_size, tile_rows, prec};
    auto builder = [](const SparseBrgemmKey& k) {
        return std::make_shared<ov::Extensions::Cpu::SparseBrgemmKernels>(k.pack,
                                                                          k.k_head_size,
                                                                          k.v_head_size,
                                                                          k.tile_rows,
                                                                          k.prec);
    };
    auto result = cache->getOrCreate(key, builder);
    OPENVINO_ASSERT(result.first, "Sparse attention brgemm kernels creation fails");
    return *result.first;
}

/// @brief The matrix-vector AMX kernel for `head_size` x `block_size` keys, created once per set of parameters.
inline std::shared_ptr<ov::Extensions::Cpu::JitMatMulVecAMX> get_sparse_gemv_kernel(const MultiCachePtr& cache,
                                                                                    int head_size,
                                                                                    int block_size,
                                                                                    ov::element::Type prec) {
    const SparseGemvKey key{head_size, block_size, prec};
    auto builder = [](const SparseGemvKey& k) {
        return std::make_shared<ov::Extensions::Cpu::JitMatMulVecAMX>(k.head_size, k.block_size, k.prec);
    };
    auto result = cache->getOrCreate(key, builder);
    OPENVINO_ASSERT(result.first, "Sparse attention AMX gemv kernel creation fails");
    return result.first;
}

}  // namespace ov::intel_cpu::node

#endif  // OPENVINO_ARCH_X86_64
