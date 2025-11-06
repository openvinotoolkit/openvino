#include "LRUCache.hpp"
#include <iostream>

LRUCache::LRUCache(size_t max_total_experts, EvictCallback cb)
    : m_max_total_experts(max_total_experts),
      m_total_experts(0),
      m_on_evict(std::move(cb)) {}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end())
        return;
    m_list.splice(m_list.end(), m_list, it);
}

LRUCache::NodeAction LRUCache::insert_or_refresh(size_t layer, size_t expert, void* addr, void* params) {
    Key key{layer, expert};
    auto it = m_map.find(key);
    if (it != m_map.end()) {
        it->second->addr = addr ? addr : it->second->addr;
        it->second->params = params ? params : it->second->params;
        move_to_end(it->second);
        return REFRESH;
    }

    if (!addr) { // call with addr == NULL to check, waiting for alloc down and update later
        return INSERT;
    }

    if (m_total_experts > m_max_total_experts) {
        evict_one();
    }

    m_list.push_back(Node{layer, expert, addr, params});
    auto new_it = std::prev(m_list.end());
    m_map[key] = new_it;
    ++m_total_experts;
    return INSERT;
}

void LRUCache::evict_one() {
    if (m_list.empty()) return;

    auto& oldest = m_list.front();
    if (m_on_evict)
        m_on_evict(oldest.layer, oldest.expert, oldest.addr, oldest.params);

    Key key{oldest.layer, oldest.expert};
    m_map.erase(key);
    m_list.pop_front();
    --m_total_experts;
}

void* LRUCache::get_expert_addr(size_t layer, size_t expert) {
    Key key{layer, expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) return nullptr;
    move_to_end(it->second);
    return it->second->addr;
}

void* LRUCache::get_expert_params(size_t layer, size_t expert) {
    Key key{layer, expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) return nullptr;
    move_to_end(it->second);
    return it->second->params;
}