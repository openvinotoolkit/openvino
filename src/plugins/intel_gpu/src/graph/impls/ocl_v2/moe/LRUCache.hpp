// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "intel_gpu/runtime/engine.hpp"

class LRUCache {
public:
    using EvictCallback = std::function<void(size_t layer, size_t expert, void* addr, void* params)>;

    enum NodeAction { INSERT, REFRESH };

    LRUCache(size_t max_total_experts, EvictCallback cb = nullptr);
    NodeAction insert_or_refresh(size_t layer, size_t expert, void* addr, void* params = nullptr);

    std::pair<size_t, bool> get_lru_item(size_t layer, size_t expert);
    size_t get_total_experts() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_total_experts;
    }

    void evict_one();

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_total_experts;
    }
    std::pair<size_t, bool> get_item(size_t layer, size_t expert);
    void set_filled(size_t lru_expert_no) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (lru_expert_no >= m_filled_list.size()) {
            return;
        }
        m_filled_list[lru_expert_no] = true;
    }

    bool m_initialized = false;

private:
    struct Key {
        size_t layer;
        size_t expert;
        bool operator==(const Key& other) const noexcept {
            return layer == other.layer && expert == other.expert;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<size_t>()(k.layer * 131ULL + k.expert);
        }
    };

    struct Node {
        size_t layer;
        size_t expert;
        size_t lru_expert_no;
    };

    size_t m_max_total_experts;
    size_t m_per_expert_size;
    size_t m_total_experts;
    size_t m_to_filled_lru_expert_no;
    EvictCallback m_on_evict;

    std::list<Node> m_list;
    std::vector<bool> m_filled_list;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;
    mutable std::mutex m_mutex;

    void move_to_end(std::list<Node>::iterator it);
    void evict_one_unlocked();
};