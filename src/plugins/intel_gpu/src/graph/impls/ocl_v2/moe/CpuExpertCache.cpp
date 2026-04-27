// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "CpuExpertCache.hpp"

namespace ov::intel_gpu::ocl {

CpuExpertCache::CpuExpertCache(size_t max_experts) : m_max_experts(max_experts), m_pool(max_experts) {
    m_free_slots.reserve(max_experts);
    for (size_t i = max_experts; i > 0; --i) {
        m_free_slots.push_back(i - 1);
    }
}

const CpuExpertCache::ExpertData* CpuExpertCache::lookup(size_t layer, size_t expert) {
    std::lock_guard<std::mutex> lock(m_mutex);
    CacheKey key{layer, expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) {
        ++m_misses;
        return nullptr;
    }
    // Move to back (MRU)
    m_lru_list.splice(m_lru_list.end(), m_lru_list, it->second);
    ++m_hits;
    return &m_pool[it->second->pool_index];
}

void CpuExpertCache::store(size_t layer, size_t expert, ExpertData&& data) {
    std::lock_guard<std::mutex> lock(m_mutex);
    CacheKey key{layer, expert};

    // If already present, update in place
    auto it = m_map.find(key);
    if (it != m_map.end()) {
        m_pool[it->second->pool_index] = std::move(data);
        m_lru_list.splice(m_lru_list.end(), m_lru_list, it->second);
        return;
    }

    // Evict if full
    if (m_free_slots.empty()) {
        auto& victim = m_lru_list.front();
        m_free_slots.push_back(victim.pool_index);
        m_map.erase(victim.key);
        m_lru_list.pop_front();
    }

    size_t slot = m_free_slots.back();
    m_free_slots.pop_back();
    m_pool[slot] = std::move(data);
    m_lru_list.push_back({key, slot});
    m_map[key] = std::prev(m_lru_list.end());
}

size_t CpuExpertCache::size() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_map.size();
}

}  // namespace ov::intel_gpu::ocl
