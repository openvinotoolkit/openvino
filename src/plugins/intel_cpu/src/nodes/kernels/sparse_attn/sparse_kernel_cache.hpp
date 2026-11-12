// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Kernel cache of the sparse attention nodes: the brgemm kernels depend only on the shapes and the data type, so nodes
// with the same parameters (e.g. the attention layers of a model) share them through the graph context's params cache.

#ifdef OPENVINO_ARCH_X86_64

#    include <common/utils.hpp>
#    include <cstddef>
#    include <memory>

#    include "cache/multi_cache.h"
#    include "nodes/kernels/sparse_attn/sparse_brgemm.hpp"
#    include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

/// @brief Parameters of the brgemm kernels (see SparseBrgemmKernels).
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

/// @brief The brgemm kernels for the pack of query heads, created once per set of parameters. The
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

}  // namespace ov::intel_cpu::node

#endif  // OPENVINO_ARCH_X86_64
