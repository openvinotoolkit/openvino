#pragma once
#include <functional>
#include <unordered_map>
#include <list>
#include <utility>

class LRUCache {
public:
    using EvictCallback = std::function<void(size_t layer, size_t expert, void* addr, void* params)>;

    enum NodeAction {
        INSERT,
        REFRESH
    };

    LRUCache(size_t max_total_experts, EvictCallback cb = nullptr);

    NodeAction insert_or_refresh(size_t layer, size_t expert, void* addr, void* params = nullptr);

    void* get_expert_addr(size_t layer, size_t expert);
    void* get_expert_params(size_t layer, size_t expert);

    void evict_one();

    size_t size() const { return m_total_experts; }

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
        void* addr;
        void* params;
    };

    size_t m_max_total_experts;
    size_t m_total_experts;
    EvictCallback m_on_evict;

    std::list<Node> m_list;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;

    void move_to_end(std::list<Node>::iterator it);
};