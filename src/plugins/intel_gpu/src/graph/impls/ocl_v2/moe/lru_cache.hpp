// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ov::intel_gpu::ocl::moe {

class LRUCache {
public:
    explicit LRUCache(size_t max_total_experts);

    std::pair<size_t, bool> get_lru_item(size_t expert);

    void evict_one();

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_total_experts;
    }

    void set_filled(size_t lru_expert_no) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (lru_expert_no >= m_filled_list.size()) {
            return;
        }
        m_filled_list[lru_expert_no] = true;
    }

    bool is_initialized() const { return m_initialized; }
    void set_initialized() { m_initialized = true; }

private:
    bool m_initialized = false;
    struct Key {
        size_t expert;
        bool operator==(const Key& other) const noexcept {
            return expert == other.expert;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<size_t>()(k.expert);
        }
    };

    struct Node {
        size_t expert;
        size_t lru_expert_no;
    };

    size_t m_max_total_experts;
    size_t m_total_experts;
    size_t m_to_filled_lru_expert_no;

    std::list<Node> m_list;
    std::vector<bool> m_filled_list;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;
    mutable std::mutex m_mutex;

    void move_to_end(std::list<Node>::iterator it);
    void evict_one_unlocked();
};

}  // namespace ov::intel_gpu::ocl::moe
