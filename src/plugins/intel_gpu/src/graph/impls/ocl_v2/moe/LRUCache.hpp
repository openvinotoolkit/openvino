#pragma once
#include <functional>
#include <unordered_map>
#include <list>
#include <utility>
#include "intel_gpu/runtime/engine.hpp"
#include <memory>

class LRUCache {
public:
    using EvictCallback = std::function<void(size_t layer, size_t expert, void* addr, void* params)>;

    enum NodeAction {
        INSERT,
        REFRESH
    };

    LRUCache(size_t max_total_experts, EvictCallback cb = nullptr);
    NodeAction insert_or_refresh(size_t layer, size_t expert, void* addr, void* params = nullptr);

    std::pair<size_t, bool> get_lru_item(size_t layer, size_t expert);
    size_t get_total_experts() const { return m_total_experts; }

    void evict_one();

    size_t size() const { return m_total_experts; }
    std::pair<size_t, bool> get_item(size_t layer, size_t expert);

    void set_filled(size_t lru_expert_no) {
        if (lru_expert_no >= m_filled_list.size()) {
            std::cout << "lru_expert_no should be smaller than max_total_experts!" << std::endl;
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

    void move_to_end(std::list<Node>::iterator it);
};