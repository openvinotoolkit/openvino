// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "LRUCache.hpp"

#include <cstdlib>


LRUCache::LRUCache(size_t max_total_experts, EvictCallback cb)
    : m_max_total_experts(max_total_experts),
      m_total_experts(0),
      m_to_filled_lru_expert_no(0),
      m_on_evict(std::move(cb)) {
        m_filled_list.resize(max_total_experts, false);
      }

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end())
        return;
    m_list.splice(m_list.end(), m_list, it);
}

void LRUCache::evict_one() {
    if (m_list.empty()) return;

    auto& oldest = m_list.front();

    m_filled_list[oldest.lru_expert_no] = false;
    m_to_filled_lru_expert_no = oldest.lru_expert_no;
    Key key{oldest.layer, oldest.expert};
    m_map.erase(key);
    m_list.pop_front();
    --m_total_experts;
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t layer, size_t expert) {
   Key key{layer, expert};
   auto it = m_map.find(key);
   if (it == m_map.end()) {
       size_t to_filled_no = 0; 
       if (m_total_experts >= m_max_total_experts) {
           evict_one();
           to_filled_no = m_to_filled_lru_expert_no;
       } else {
           to_filled_no = m_total_experts;
       }
       m_list.push_back(Node{layer, expert, to_filled_no});
       auto new_it = std::prev(m_list.end());
       m_map[key] = new_it;
       ++m_total_experts;
       return { to_filled_no, false };
   } else {
       move_to_end(it->second);
       const bool is_hit = m_filled_list[it->second->lru_expert_no];
       return { it->second->lru_expert_no, is_hit };
   }
}