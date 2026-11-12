// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <list>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
#include <functional>

#include "intel_gpu/runtime/event.hpp"

namespace ov::intel_gpu::ocl::moe {

class ExpertSlotReservations {
public:
    struct Ticket {
        uint64_t id;
        uint64_t epoch;
        std::vector<size_t> slots;
        std::vector<uint32_t> missing_experts;
        std::vector<size_t> missing_slots;
        std::vector<size_t> ready_pairs;
        std::vector<size_t> missing_pairs;
    };

    explicit ExpertSlotReservations(size_t capacity);
    std::optional<Ticket> reserve(const std::vector<uint32_t>& experts, uint64_t epoch, bool speculative = false);
    void discard_unused(const Ticket& ticket, const std::vector<uint32_t>& used);
    bool rollback(const Ticket& ticket);
    bool complete(const Ticket& ticket);
    bool release(const Ticket& ticket);

private:
    struct Slot {
        std::optional<uint32_t> expert;
        uint64_t generation = 0;
        uint64_t age = 0;
        size_t pins = 0;
        bool ready = false;
    };
    struct Record {
        uint64_t epoch;
        std::vector<std::pair<size_t, uint64_t>> generations;
        std::vector<std::pair<size_t, Slot>> replaced;
    };
    std::mutex _mutex;
    std::vector<Slot> _slots;
    std::unordered_map<uint64_t, Record> _active;
    uint64_t _epoch = 0;
    uint64_t _clock = 0;
    uint64_t _next_id = 0;
};

enum class SlotStatus : uint8_t { READY, PENDING_LOAD, PINNED };

struct CacheItemExtended {
    SlotStatus status = SlotStatus::READY;
    cldnn::event::ptr ready_event;
    cldnn::event::ptr last_use_event;
    uint64_t generation = 0;
    bool loading = false;
    bool demanded = false;
};

class LRUCache {
public:
    explicit LRUCache(size_t max_total_experts);

    struct Load {
        uint32_t expert_id;
        size_t slot;
        uint64_t generation;
        cldnn::event::ptr ready_event;
        cldnn::event::ptr last_use_event;
    };

    struct Partition {
        std::vector<size_t> slots;
        std::vector<size_t> ready_pairs;
        std::vector<size_t> inflight_pairs;
        std::vector<size_t> missing_pairs;
        std::vector<Load> missing;
        std::vector<cldnn::event::ptr> ready_events;
        std::vector<cldnn::event::ptr> wait_events;
    };

    using EventFactory = std::function<cldnn::event::ptr()>;
    std::optional<Partition> partition_3way(const std::vector<uint32_t>& topk_ids,
                                          const EventFactory& make_event,
                                          bool allow_pending_reuse = false);
    std::optional<Load> reserve_speculative(uint32_t expert_id, size_t top_k, const EventFactory& make_event);
    bool complete_load(const Load& load);
    bool cancel_load(const Load& load);
    std::optional<Load> get_pending_load(uint32_t expert_id) const;
    void unpin_all(const cldnn::event::ptr& complete_event);

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

    bool is_initialized() const {
        return m_initialized;
    }
    void set_initialized() {
        m_initialized = true;
    }

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
        CacheItemExtended state{};
    };

    size_t m_max_total_experts;
    size_t m_total_experts;
    size_t m_to_filled_lru_expert_no;

    std::list<Node> m_list;
    std::vector<bool> m_filled_list;
    std::vector<cldnn::event::ptr> m_slot_last_use;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;
    mutable std::mutex m_mutex;
    uint64_t m_generation = 0;

    void move_to_end(std::list<Node>::iterator it);
    void evict_one_unlocked();
};

}  // namespace ov::intel_gpu::ocl::moe
