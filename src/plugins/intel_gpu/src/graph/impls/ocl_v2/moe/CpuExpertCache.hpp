// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::ocl {

// CPU-side L2 cache for MoE expert weight data.
// Sits between GPU LRU cache and disk, caching post-transpose tensor
// data in host memory so that GPU cache misses can skip disk I/O.
class CpuExpertCache {
public:
    static constexpr size_t tensor_count = cldnn::moe_3gemm_fused_compressed::serialized_weight_offset_count;  // 9

    struct ExpertData {
        std::array<std::vector<uint8_t>, tensor_count> tensors;  // post-transpose data
        std::array<size_t, tensor_count> sizes{};
    };

    explicit CpuExpertCache(size_t max_experts);

    // Returns pointer to cached data on hit, nullptr on miss.
    // Thread-safe. On hit, promotes the entry to most-recently-used.
    const ExpertData* lookup(size_t layer, size_t expert);

    // Store expert data. Evicts LRU entry if cache is full.
    // Thread-safe. Takes ownership of data via move.
    void store(size_t layer, size_t expert, ExpertData&& data);

    size_t size() const;
    size_t capacity() const { return m_max_experts; }

    uint64_t hits() const { return m_hits; }
    uint64_t misses() const { return m_misses; }

private:
    struct CacheKey {
        size_t layer;
        size_t expert;
        bool operator==(const CacheKey& o) const { return layer == o.layer && expert == o.expert; }
    };

    struct KeyHash {
        size_t operator()(const CacheKey& k) const {
            return std::hash<size_t>()(k.layer) ^ (std::hash<size_t>()(k.expert) << 16);
        }
    };

    struct Node {
        CacheKey key;
        size_t pool_index;
    };

    size_t m_max_experts;
    std::list<Node> m_lru_list;  // front = LRU, back = MRU
    std::unordered_map<CacheKey, std::list<Node>::iterator, KeyHash> m_map;
    std::vector<ExpertData> m_pool;
    std::vector<size_t> m_free_slots;
    mutable std::mutex m_mutex;
    uint64_t m_hits = 0;
    uint64_t m_misses = 0;
};

}  // namespace ov::intel_gpu::ocl
