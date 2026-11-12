// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lru_cache.hpp"
#include <algorithm>
#include <unordered_set>
#include "openvino/core/except.hpp"

namespace ov::intel_gpu::ocl::moe {

ExpertSlotReservations::ExpertSlotReservations(size_t capacity) : _slots(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("LES requires at least one resident slot");
    }
}

std::optional<ExpertSlotReservations::Ticket> ExpertSlotReservations::reserve(const std::vector<uint32_t>& experts, uint64_t epoch, bool speculative) {
    std::lock_guard<std::mutex> lock(_mutex);
    std::unordered_set<uint32_t> unique(experts.begin(), experts.end());
    if (unique.size() > _slots.size() || epoch < _epoch || (epoch != _epoch && !_active.empty())) {
        return std::nullopt;
    }
    auto planned = _slots;
    std::unordered_map<uint32_t, size_t> mapping;
    Record record{epoch, {}, {}};
    Ticket result{_next_id + 1, epoch, {}, {}, {}, {}, {}};
    auto clock = _clock;
    for (auto expert : unique) {
        for (size_t index = 0; index < planned.size(); ++index) {
            auto& slot = planned[index];
            if (slot.expert == expert) {
                if (!slot.ready) {
                    return std::nullopt;
                }
                ++slot.pins;
                if (!speculative) {
                    slot.age = ++clock;
                }
                mapping.emplace(expert, index);
                record.generations.emplace_back(index, slot.generation);
                break;
            }
        }
    }
    for (auto expert : experts) {
        if (mapping.count(expert)) {
            continue;
        }
        size_t candidate = planned.size();
        for (size_t index = 0; index < planned.size(); ++index) {
            const auto& slot = planned[index];
            if (slot.pins == 0 &&
                (candidate == planned.size() || !slot.expert || slot.age < planned[candidate].age)) {
                candidate = index;
                if (!slot.expert) {
                    break;
                }
            }
        }
        if (candidate == planned.size()) {
            return std::nullopt;
        }
        auto& slot = planned[candidate];
        if (speculative) {
            record.replaced.emplace_back(candidate, slot);
        }
        slot.expert = expert;
        ++slot.generation;
        slot.age = ++clock;
        slot.pins = 1;
        slot.ready = false;
        mapping.emplace(expert, candidate);
        record.generations.emplace_back(candidate, slot.generation);
        result.missing_experts.push_back(expert);
        result.missing_slots.push_back(candidate);
    }
    for (size_t pair = 0; pair < experts.size(); ++pair) {
        auto slot = mapping.at(experts[pair]);
        result.slots.push_back(slot);
        (planned[slot].ready ? result.ready_pairs : result.missing_pairs).push_back(pair);
    }
    _active.emplace(result.id, std::move(record));
    _slots.swap(planned);
    _epoch = epoch;
    _clock = clock;
    ++_next_id;
    return result;
}

void ExpertSlotReservations::discard_unused(const Ticket& ticket, const std::vector<uint32_t>& used) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return;
    }
    for (size_t index = 0; index < ticket.missing_slots.size(); ++index) {
        auto slot_index = ticket.missing_slots[index];
        auto generation = std::find_if(record->second.generations.begin(), record->second.generations.end(),
                                       [&](const auto& entry) { return entry.first == slot_index; });
        auto& slot = _slots[slot_index];
        if (generation != record->second.generations.end() && slot.generation == generation->second && slot.pins == 1 &&
            std::find(used.begin(), used.end(), ticket.missing_experts[index]) == used.end()) {
            slot.ready = false;
        }
    }
}

bool ExpertSlotReservations::rollback(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        if (_slots[entry.first].generation != entry.second || _slots[entry.first].pins != 1) {
            return false;
        }
    }
    for (const auto& entry : record->second.generations) {
        --_slots[entry.first].pins;
    }
    for (const auto& entry : record->second.replaced) {
        _slots[entry.first] = entry.second;
    }
    _active.erase(record);
    return true;
}

bool ExpertSlotReservations::complete(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        if (_slots[entry.first].generation != entry.second) {
            return false;
        }
    }
    for (const auto& entry : record->second.generations) {
        _slots[entry.first].ready = true;
    }
    return true;
}

bool ExpertSlotReservations::release(const Ticket& ticket) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto record = _active.find(ticket.id);
    if (record == _active.end() || record->second.epoch != ticket.epoch) {
        return false;
    }
    for (const auto& entry : record->second.generations) {
        auto& slot = _slots[entry.first];
        --slot.pins;
        if (!slot.ready && slot.pins == 0) {
            slot.expert.reset();
        }
    }
    _active.erase(record);
    return true;
}

LRUCache::LRUCache(size_t max_total_experts) : m_max_total_experts(max_total_experts), m_total_experts(0), m_to_filled_lru_expert_no(0) {
    m_filled_list.resize(max_total_experts, false);
    m_slot_last_use.resize(max_total_experts);
}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end()) {
        return;
    }
    m_list.splice(m_list.end(), m_list, it);
}

std::optional<LRUCache::Partition> LRUCache::partition_3way(const std::vector<uint32_t>& topk_ids,
                                                        const EventFactory& make_event,
                                                        bool allow_pending_reuse) {
    std::lock_guard<std::mutex> lock(m_mutex);
    const std::unordered_set<uint32_t> requested(topk_ids.begin(), topk_ids.end());
    if (requested.size() > m_max_total_experts) {
        return std::nullopt;
    }
    for (auto expert : requested) {
        auto found = m_map.find(Key{expert});
        OPENVINO_ASSERT(found == m_map.end() || found->second->state.loading ||
                            m_filled_list[found->second->lru_expert_no],
                        "Lookahead cannot consume an unfilled OTD slot");
    }
    Partition result;
    std::unordered_set<uint32_t> missing;
    std::vector<bool> occupied(m_max_total_experts, false);
    for (const auto& node : m_list) {
        occupied[node.lru_expert_no] = true;
    }
    std::vector<std::list<Node>::iterator> victims;
    for (auto expert : topk_ids) {
        if (m_map.count(Key{expert}) || !missing.insert(expert).second) {
            continue;
        }
        auto free_slot = std::find(occupied.begin(), occupied.end(), false);
        size_t slot = static_cast<size_t>(free_slot - occupied.begin());
        cldnn::event::ptr last_use;
        if (free_slot == occupied.end()) {
            auto victim = std::find_if(m_list.begin(), m_list.end(), [&](const Node& node) {
                return !requested.count(static_cast<uint32_t>(node.expert)) &&
                                             ((node.state.status == SlotStatus::READY && !node.state.loading) ||
                                                (allow_pending_reuse && node.state.status == SlotStatus::PENDING_LOAD &&
                                                 node.state.demanded && node.state.last_use_event)) &&
                       std::none_of(victims.begin(), victims.end(), [&](const auto& chosen) { return &*chosen == &node; });
            });
            if (victim == m_list.end()) {
                return std::nullopt;
            }
            victims.push_back(victim);
            slot = victim->lru_expert_no;
            last_use = victim->state.last_use_event;
        } else {
            occupied[slot] = true;
            last_use = m_slot_last_use[slot];
        }
        result.missing.push_back({expert, slot, 0, {}, last_use});
    }
    for (auto& load : result.missing) {
        load.ready_event = make_event();
        OPENVINO_ASSERT(load.ready_event, "Lookahead load requires a completion event");
    }
    for (auto victim : victims) {
        m_map.erase(Key{victim->expert});
        m_list.erase(victim);
        --m_total_experts;
    }
    for (auto& load : result.missing) {
        load.generation = ++m_generation;
        m_list.push_back(Node{load.expert_id, load.slot,
                             {SlotStatus::PENDING_LOAD, load.ready_event, load.last_use_event, load.generation, true}});
        m_map.emplace(Key{load.expert_id}, std::prev(m_list.end()));
        m_filled_list[load.slot] = false;
        ++m_total_experts;
    }
    std::unordered_set<uint32_t> waiting;
    for (size_t pair = 0; pair < topk_ids.size(); ++pair) {
        const auto expert = topk_ids[pair];
        auto node = m_map.at(Key{expert});
        result.slots.push_back(node->lru_expert_no);
        if (missing.count(expert)) {
            result.missing_pairs.push_back(pair);
        } else if (node->state.loading) {
            result.inflight_pairs.push_back(pair);
        } else {
            OPENVINO_ASSERT(m_filled_list[node->lru_expert_no], "Lookahead cannot consume an unfilled OTD slot");
            result.ready_pairs.push_back(pair);
            if (node->state.ready_event && waiting.insert(expert).second) {
                result.ready_events.push_back(node->state.ready_event);
            }
        }
        if (node->state.loading && waiting.insert(expert).second) {
            result.wait_events.push_back(node->state.ready_event);
        }
        node->state.status = SlotStatus::PINNED;
        node->state.demanded = true;
        move_to_end(node);
    }
    return result;
}

std::optional<LRUCache::Load> LRUCache::reserve_speculative(uint32_t expert_id,
                                                         size_t top_k,
                                                         const EventFactory& make_event) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (top_k >= m_max_total_experts || m_map.count(Key{expert_id})) {
        return std::nullopt;
    }
    const auto quota = std::min({top_k, m_max_total_experts / 4, m_max_total_experts - top_k});
    const auto pending = static_cast<size_t>(std::count_if(m_list.begin(), m_list.end(), [](const Node& node) {
        return node.state.loading;
    }));
    if (pending >= quota) {
        return std::nullopt;
    }
    std::vector<bool> occupied(m_max_total_experts, false);
    for (const auto& node : m_list) {
        occupied[node.lru_expert_no] = true;
    }
    auto free_slot = std::find(occupied.begin(), occupied.end(), false);
    size_t slot = static_cast<size_t>(free_slot - occupied.begin());
    auto victim = m_list.end();
    cldnn::event::ptr last_use;
    if (free_slot == occupied.end()) {
        victim = std::find_if(m_list.begin(), m_list.end(), [](const Node& node) {
            return node.state.status == SlotStatus::READY && !node.state.loading;
        });
        if (victim == m_list.end()) {
            return std::nullopt;
        }
        slot = victim->lru_expert_no;
        last_use = victim->state.last_use_event;
    }
    auto ready = make_event();
    if (victim == m_list.end()) {
        last_use = m_slot_last_use[slot];
    }
    OPENVINO_ASSERT(ready, "Lookahead load requires a completion event");
    if (victim != m_list.end()) {
        m_map.erase(Key{victim->expert});
        m_list.erase(victim);
        --m_total_experts;
    }
    Load load{expert_id, slot, ++m_generation, ready, last_use};
    m_list.push_back(Node{expert_id, slot, {SlotStatus::PENDING_LOAD, ready, last_use, load.generation, true}});
    m_map.emplace(Key{expert_id}, std::prev(m_list.end()));
    m_filled_list[slot] = false;
    ++m_total_experts;
    return load;
}

std::optional<LRUCache::Load> LRUCache::get_pending_load(uint32_t expert_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{expert_id});
    if (found == m_map.end() || !found->second->state.loading) {
        return std::nullopt;
    }
    const auto& node = *found->second;
    return Load{expert_id, node.lru_expert_no, node.state.generation, node.state.ready_event, node.state.last_use_event};
}

bool LRUCache::complete_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation) {
        return false;
    }
    auto& node = *found->second;
    node.state.loading = false;
    if (node.state.status != SlotStatus::PINNED) {
        node.state.status = SlotStatus::READY;
    }
    m_filled_list[node.lru_expert_no] = true;
    return true;
}

bool LRUCache::cancel_load(const Load& load) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto found = m_map.find(Key{load.expert_id});
    if (found == m_map.end() || found->second->state.generation != load.generation ||
        found->second->state.demanded || !found->second->state.loading) {
        return false;
    }
    m_filled_list[found->second->lru_expert_no] = false;
    m_list.erase(found->second);
    m_map.erase(found);
    --m_total_experts;
    return true;
}

void LRUCache::unpin_all(const cldnn::event::ptr& complete_event) {
    OPENVINO_ASSERT(complete_event, "Lookahead unpin requires a GPU consumer event");
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& node : m_list) {
        if (node.state.status == SlotStatus::PINNED) {
            node.state.last_use_event = complete_event;
            m_slot_last_use[node.lru_expert_no] = complete_event;
            node.state.status = node.state.loading ? SlotStatus::PENDING_LOAD : SlotStatus::READY;
        }
    }
}

void LRUCache::evict_one_unlocked() {
    if (m_list.empty()) {
        return;
    }

    auto candidate = std::find_if(m_list.begin(), m_list.end(), [](const Node& node) {
        return node.state.status == SlotStatus::READY && !node.state.loading;
    });
    OPENVINO_ASSERT(candidate != m_list.end(), "Cannot evict a pinned or pending lookahead slot");
    auto& oldest = *candidate;

    m_filled_list[oldest.lru_expert_no] = false;
    m_to_filled_lru_expert_no = oldest.lru_expert_no;
    Key key{oldest.expert};
    m_map.erase(key);
    m_list.erase(candidate);
    --m_total_experts;
}

void LRUCache::evict_one() {
    std::lock_guard<std::mutex> lock(m_mutex);
    evict_one_unlocked();
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t expert) {
    std::lock_guard<std::mutex> lock(m_mutex);

    Key key{expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) {
        size_t to_filled_no = 0;
        if (m_total_experts >= m_max_total_experts) {
            evict_one_unlocked();
            to_filled_no = m_to_filled_lru_expert_no;
        } else {
            std::vector<bool> occupied(m_max_total_experts, false);
            for (const auto& node : m_list) {
                occupied[node.lru_expert_no] = true;
            }
            to_filled_no = static_cast<size_t>(std::find(occupied.begin(), occupied.end(), false) - occupied.begin());
        }
        m_list.push_back(Node{expert, to_filled_no});
        auto new_it = std::prev(m_list.end());
        m_map[key] = new_it;
        ++m_total_experts;
        return {to_filled_no, false};
    }
    OPENVINO_ASSERT(!it->second->state.loading && it->second->state.status != SlotStatus::PINNED,
                    "Synchronous OTD cannot acquire an active lookahead slot");
    move_to_end(it->second);
    const bool is_hit = m_filled_list[it->second->lru_expert_no];
    return {it->second->lru_expert_no, is_hit};
}

}  // namespace ov::intel_gpu::ocl::moe
