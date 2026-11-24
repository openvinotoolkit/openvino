// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "ocl_v2/moe/lru_cache.hpp"

#include <thread>
#include <vector>
#include <set>
#include <atomic>

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::LRUCache;
using ov::intel_gpu::ocl::moe::ExpertSlotReservations;

namespace {
class LookaheadTestEvent : public cldnn::event {
    void wait_impl() override { ADD_FAILURE() << "LRU operations must not wait for GPU events"; }
    void set_impl() override {}
    bool is_set_impl() override { return false; }
};

cldnn::event::ptr make_lookahead_test_event() {
    return std::make_shared<LookaheadTestEvent>();
}
}

TEST(moe_lookahead_lru, partitions_and_pins_ready_inflight_missing_atomically) {
    LRUCache cache(8);
    const auto resident = cache.get_lru_item(1);
    cache.set_filled(resident.first);
    const auto speculative = cache.reserve_speculative(2, 3, make_lookahead_test_event);
    ASSERT_TRUE(speculative);
    EXPECT_FALSE(cache.reserve_speculative(2, 3, make_lookahead_test_event));
    const auto demand = cache.partition_3way({1, 2, 3, 2}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    EXPECT_EQ(demand->ready_pairs, (std::vector<size_t>{0}));
    EXPECT_EQ(demand->inflight_pairs, (std::vector<size_t>{1, 3}));
    EXPECT_EQ(demand->missing_pairs, (std::vector<size_t>{2}));
    ASSERT_EQ(demand->missing.size(), 1U);
    EXPECT_EQ(demand->missing.front().expert_id, 3U);
    EXPECT_EQ(demand->slots[1], speculative->slot);
    EXPECT_EQ(demand->slots[1], demand->slots[3]);
    ASSERT_EQ(demand->wait_events.size(), 2U);
    EXPECT_EQ(demand->wait_events.front(), speculative->ready_event);
    EXPECT_FALSE(cache.cancel_load(*speculative));
    cache.unpin_all(make_lookahead_test_event());
    EXPECT_FALSE(cache.cancel_load(*speculative));
    EXPECT_TRUE(cache.complete_load(*speculative));
    EXPECT_TRUE(cache.complete_load(demand->missing.front()));
    const auto hits = cache.partition_3way({1, 2, 3}, make_lookahead_test_event);
    ASSERT_TRUE(hits);
    EXPECT_EQ(hits->ready_pairs.size(), 3U);
    EXPECT_TRUE(hits->wait_events.empty());
    ASSERT_EQ(hits->ready_events.size(), 2U);
    EXPECT_EQ(hits->ready_events.front(), speculative->ready_event);
}

TEST(moe_lookahead_lru, preserves_war_event_on_slot_replacement) {
    LRUCache cache(1);
    auto first = cache.partition_3way({1}, make_lookahead_test_event);
    ASSERT_TRUE(first);
    ASSERT_TRUE(cache.complete_load(first->missing.front()));
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event));
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    auto next = cache.partition_3way({2}, make_lookahead_test_event);
    ASSERT_TRUE(next);
    ASSERT_EQ(next->missing.size(), 1U);
    EXPECT_EQ(next->missing.front().last_use_event, consumer);
    EXPECT_EQ(next->missing.front().slot, first->missing.front().slot);
    EXPECT_FALSE(cache.complete_load(first->missing.front()));
}

TEST(moe_lookahead_lru, pending_waves_reuse_slots_only_after_consumer_is_recorded) {
    LRUCache cache(1);
    auto first = cache.partition_3way({1}, make_lookahead_test_event);
    ASSERT_TRUE(first);
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event, true));
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    EXPECT_FALSE(cache.partition_3way({2}, make_lookahead_test_event));
    auto next = cache.partition_3way({2}, make_lookahead_test_event, true);
    ASSERT_TRUE(next);
    EXPECT_EQ(next->slots, first->slots);
    EXPECT_EQ(next->missing.front().last_use_event, consumer);
    EXPECT_FALSE(cache.complete_load(first->missing.front()));
    auto pending = cache.get_pending_load(2);
    ASSERT_TRUE(pending);
    EXPECT_EQ(pending->generation, next->missing.front().generation);
    EXPECT_TRUE(cache.complete_load(next->missing.front()));
}

TEST(moe_lookahead_lru, speculative_quota_reserves_demand_headroom) {
    LRUCache full(4);
    EXPECT_FALSE(full.reserve_speculative(0, 4, make_lookahead_test_event));
    LRUCache cache(8);
    ASSERT_TRUE(cache.reserve_speculative(0, 7, make_lookahead_test_event));
    EXPECT_FALSE(cache.reserve_speculative(1, 7, make_lookahead_test_event));
    auto demand = cache.partition_3way({1, 2, 3, 4, 5, 6, 7}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    EXPECT_EQ(demand->missing.size(), 7U);
}

TEST(moe_lookahead_lru, cancelled_replacement_retains_physical_slot_war) {
    LRUCache cache(4);
    auto initial = cache.partition_3way({1, 2, 3, 4}, make_lookahead_test_event);
    ASSERT_TRUE(initial);
    for (const auto& load : initial->missing) {
        ASSERT_TRUE(cache.complete_load(load));
    }
    auto consumer = make_lookahead_test_event();
    cache.unpin_all(consumer);
    auto speculative = cache.reserve_speculative(5, 1, make_lookahead_test_event);
    ASSERT_TRUE(speculative);
    ASSERT_TRUE(cache.cancel_load(*speculative));
    auto replacement = cache.partition_3way({6}, make_lookahead_test_event);
    ASSERT_TRUE(replacement);
    ASSERT_EQ(replacement->missing.size(), 1U);
    EXPECT_EQ(replacement->missing.front().slot, speculative->slot);
    EXPECT_EQ(replacement->missing.front().last_use_event, consumer);
}

TEST(moe_lookahead_lru, cancellation_and_failed_reservation_do_not_alias_slots) {
    LRUCache cache(8);
    auto cancelled = cache.reserve_speculative(1, 2, make_lookahead_test_event);
    ASSERT_TRUE(cancelled);
    ASSERT_TRUE(cache.reserve_speculative(2, 2, make_lookahead_test_event));
    ASSERT_TRUE(cache.cancel_load(*cancelled));
    EXPECT_FALSE(cache.complete_load(*cancelled));
    EXPECT_FALSE(cache.partition_3way({0, 1, 2, 3, 4, 5, 6, 7, 8}, make_lookahead_test_event));
    EXPECT_EQ(cache.size(), 1U);
    auto demand = cache.partition_3way({3, 2}, make_lookahead_test_event);
    ASSERT_TRUE(demand);
    EXPECT_NE(demand->slots[0], demand->slots[1]);
}

TEST(moe_les_reservations, deduplicates_and_rejects_capacity_without_mutation) {
    ExpertSlotReservations cache(2);
    EXPECT_FALSE(cache.reserve({1, 2, 3}, 1));
    auto lease = cache.reserve({1, 1, 2}, 1);
    ASSERT_TRUE(lease);
    EXPECT_EQ(lease->slots[0], lease->slots[1]);
    EXPECT_EQ(lease->missing_experts.size(), 2U);
    EXPECT_TRUE(cache.complete(*lease));
    EXPECT_TRUE(cache.release(*lease));
    auto hits = cache.reserve({1, 2}, 2);
    ASSERT_TRUE(hits);
    EXPECT_TRUE(hits->missing_experts.empty());
}

TEST(moe_les_reservations, protects_hits_before_assigning_misses) {
    ExpertSlotReservations cache(2);
    auto initial = cache.reserve({1, 2}, 1).value();
    cache.complete(initial);
    cache.release(initial);
    auto lease = cache.reserve({3, 1}, 1).value();
    EXPECT_EQ(lease.slots[1], initial.slots[0]);
    EXPECT_EQ(lease.missing_experts, std::vector<uint32_t>{3});
    EXPECT_NE(lease.slots[0], lease.slots[1]);
}

TEST(moe_les_reservations, pinned_consumers_prevent_eviction) {
    ExpertSlotReservations cache(1);
    auto first = cache.reserve({1}, 1).value();
    cache.complete(first);
    auto second = cache.reserve({1}, 1).value();
    EXPECT_FALSE(cache.reserve({2}, 1));
    cache.release(first);
    EXPECT_FALSE(cache.reserve({2}, 1));
    cache.release(second);
    EXPECT_TRUE(cache.reserve({2}, 1));
}

TEST(moe_les_reservations, pending_is_not_ready_and_abort_permits_retry) {
    ExpertSlotReservations cache(1);
    auto first = cache.reserve({1}, 1).value();
    EXPECT_FALSE(cache.reserve({1}, 1));
    EXPECT_FALSE(cache.reserve({2}, 2));
    cache.release(first);
    auto next = cache.reserve({2}, 2).value();
    EXPECT_FALSE(cache.complete(first));
    EXPECT_FALSE(cache.release(first));
    EXPECT_FALSE(cache.reserve({2}, 1));
    EXPECT_TRUE(cache.complete(next));
    EXPECT_TRUE(cache.release(next));
}

TEST(moe_les_reservations, rejects_zero_capacity) {
    EXPECT_ANY_THROW(ExpertSlotReservations(0));
}

TEST(moe_les_reservations, partitions_all_top8_masks_in_canonical_pair_order) {
    const std::vector<uint32_t> experts{7, 0, 5, 2, 6, 1, 4, 3, 3, 4, 1, 6, 2, 5, 0, 7};
    for (uint32_t mask = 0; mask < 256; ++mask) {
        SCOPED_TRACE(mask);
        ExpertSlotReservations cache(8);
        std::vector<uint32_t> warm;
        for (uint32_t expert = 0; expert < 8; ++expert) {
            if (mask & (1U << expert)) {
                warm.push_back(expert);
            }
        }
        auto initial = cache.reserve(warm, 1);
        ASSERT_TRUE(initial);
        ASSERT_TRUE(cache.complete(*initial));
        ASSERT_TRUE(cache.release(*initial));
        auto demand = cache.reserve(experts, 2);
        ASSERT_TRUE(demand);
        std::vector<size_t> ready;
        std::vector<size_t> missing;
        for (size_t pair = 0; pair < experts.size(); ++pair) {
            (mask & (1U << experts[pair]) ? ready : missing).push_back(pair);
            EXPECT_EQ(demand->slots[pair], demand->slots[experts.size() - 1 - pair]);
        }
        EXPECT_EQ(demand->ready_pairs, ready);
        EXPECT_EQ(demand->missing_pairs, missing);
        ASSERT_TRUE(cache.complete(*demand));
        EXPECT_EQ(demand->ready_pairs, ready);
        EXPECT_EQ(demand->missing_pairs, missing);
        ASSERT_TRUE(cache.release(*demand));
        auto hits = cache.reserve(experts, 3);
        ASSERT_TRUE(hits);
        EXPECT_EQ(hits->ready_pairs.size(), experts.size());
        EXPECT_TRUE(hits->missing_pairs.empty());
        EXPECT_TRUE(hits->missing_experts.empty());
        ASSERT_TRUE(cache.release(*hits));
    }
}

// --------------------------------------------------
// Basic construction and initial state
// --------------------------------------------------

TEST(moe_lru_cache, initial_state) {
    LRUCache cache(4);

    ASSERT_EQ(cache.size(), 0U);
    ASSERT_FALSE(cache.is_initialized());
}

// --------------------------------------------------
// get_lru_item: insert (miss) and hit
// --------------------------------------------------

TEST(moe_lru_cache, single_insert_is_miss) {
    LRUCache cache(4);

    auto [slot, hit] = cache.get_lru_item(/*expert=*/0);
    ASSERT_FALSE(hit);        // first access is a miss
    ASSERT_EQ(slot, 0U);     // first slot assigned
    ASSERT_EQ(cache.size(), 1U);
}

TEST(moe_lru_cache, second_access_without_fill_is_miss) {
    // get_lru_item returns is_hit based on m_filled_list, not just presence
    LRUCache cache(4);

    cache.get_lru_item(0);  // insert, slot 0
    auto [slot, hit] = cache.get_lru_item(0);  // access again

    ASSERT_EQ(slot, 0U);
    ASSERT_FALSE(hit);  // filled_list[0] is still false
}

TEST(moe_lru_cache, access_after_set_filled_is_hit) {
    LRUCache cache(4);

    auto [slot, miss] = cache.get_lru_item(0);
    cache.set_filled(slot);

    auto [slot2, hit] = cache.get_lru_item(0);
    ASSERT_EQ(slot2, slot);
    ASSERT_TRUE(hit);
}

// --------------------------------------------------
// Sequential slot allocation
// --------------------------------------------------

TEST(moe_lru_cache, slots_assigned_sequentially) {
    const size_t cap = 4;
    LRUCache cache(cap);

    for (size_t i = 0; i < cap; ++i) {
        auto [slot, hit] = cache.get_lru_item(/*expert=*/i);
        ASSERT_EQ(slot, i);
        ASSERT_FALSE(hit);
    }
    ASSERT_EQ(cache.size(), cap);
}

// --------------------------------------------------
// Eviction: oldest entry evicted when cache is full
// --------------------------------------------------

TEST(moe_lru_cache, eviction_when_full) {
    LRUCache cache(3);

    // Fill cache: experts 0, 1, 2
    cache.get_lru_item(0);  // slot 0
    cache.get_lru_item(1);  // slot 1
    cache.get_lru_item(2);  // slot 2
    ASSERT_EQ(cache.size(), 3U);

    // Insert expert 3 -> evicts expert 0 (oldest), reuses slot 0
    auto [slot, hit] = cache.get_lru_item(3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 0U);   // expert 0's slot is recycled
    ASSERT_EQ(cache.size(), 3U);  // size stays at capacity
}

TEST(moe_lru_cache, evicted_entry_becomes_miss) {
    LRUCache cache(2);

    // Fill: expert 0 (slot 0), expert 1 (slot 1)
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // Insert expert 2 -> evicts expert 0
    cache.get_lru_item(2);

    // Access expert 0 again -> should be a miss (it was evicted)
    auto [slot, hit] = cache.get_lru_item(0);
    ASSERT_FALSE(hit);
}

// --------------------------------------------------
// LRU ordering: recently accessed items survive eviction
// --------------------------------------------------

TEST(moe_lru_cache, lru_order_refresh_on_access) {
    LRUCache cache(3);

    // Insert experts 0, 1, 2 in order
    cache.get_lru_item(0);  // slot 0  (LRU order: 0)
    cache.get_lru_item(1);  // slot 1  (LRU order: 0, 1)
    cache.get_lru_item(2);  // slot 2  (LRU order: 0, 1, 2)

    // Access expert 0 -> moves to most recent
    // (LRU order: 1, 2, 0)
    cache.get_lru_item(0);

    // Insert expert 3 -> should evict expert 1 (now the oldest)
    auto [slot, hit] = cache.get_lru_item(3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 1U);  // reuses expert 1's slot
}

TEST(moe_lru_cache, double_refresh_changes_eviction_order) {
    LRUCache cache(3);

    // Insert 0, 1, 2
    cache.get_lru_item(0);  // slot 0
    cache.get_lru_item(1);  // slot 1
    cache.get_lru_item(2);  // slot 2

    // Refresh 0, then refresh 1  -> LRU order: 2, 0, 1
    cache.get_lru_item(0);
    cache.get_lru_item(1);

    // Insert 3 -> evicts expert 2 (oldest)
    auto [slot3, hit3] = cache.get_lru_item(3);
    ASSERT_FALSE(hit3);
    ASSERT_EQ(slot3, 2U);  // reuses expert 2's slot

    // Insert 4 -> evicts expert 0 (now oldest)
    auto [slot4, hit4] = cache.get_lru_item(4);
    ASSERT_FALSE(hit4);
    ASSERT_EQ(slot4, 0U);  // reuses expert 0's slot
}

// --------------------------------------------------
// Multi-layer: (layer, expert) pairs are independent
// --------------------------------------------------

// Since LRU cache is per-layer (each MoE layer has its own cache instance),
// the layer dimension has been removed from the cache key. This test verifies
// that distinct expert ids get independent slots.
TEST(moe_lru_cache, distinct_experts_get_independent_slots) {
    LRUCache cache(4);

    auto [s0, h0] = cache.get_lru_item(/*expert=*/0);
    auto [s1, h1] = cache.get_lru_item(/*expert=*/1);

    ASSERT_FALSE(h0);
    ASSERT_FALSE(h1);
    ASSERT_NE(s0, s1);  // different slots for different experts
    ASSERT_EQ(cache.size(), 2U);
}

TEST(moe_lru_cache, same_expert_gets_same_slot) {
    LRUCache cache(4);

    cache.get_lru_item(5);
    cache.set_filled(0);

    auto [s0, h0] = cache.get_lru_item(5);

    ASSERT_TRUE(h0);
    ASSERT_EQ(s0, 0U);
}

// --------------------------------------------------
// set_filled / filled tracking
// --------------------------------------------------

TEST(moe_lru_cache, filled_cleared_on_eviction) {
    LRUCache cache(2);

    // Insert expert 0, mark filled
    auto [s0, miss0] = cache.get_lru_item(0);
    cache.set_filled(s0);

    // Insert expert 1
    cache.get_lru_item(1);

    // Insert expert 2 -> evicts expert 0, reuses slot s0
    auto [s2, h2] = cache.get_lru_item(2);
    ASSERT_EQ(s2, s0);         // recycled slot
    ASSERT_FALSE(h2);          // filled was cleared during eviction

    // Even after re-inserting to the same slot, accessing it returns not-filled
    auto [s2b, h2b] = cache.get_lru_item(2);
    ASSERT_FALSE(h2b);  // not filled until set_filled is called
}

TEST(moe_lru_cache, set_filled_out_of_range_is_safe) {
    LRUCache cache(2);
    // Should not crash
    cache.set_filled(100);
    cache.set_filled(std::numeric_limits<size_t>::max());
}

// --------------------------------------------------
// evict_one: explicit eviction
// --------------------------------------------------

TEST(moe_lru_cache, explicit_evict_reduces_size) {
    LRUCache cache(4);

    cache.get_lru_item(0);
    cache.get_lru_item(1);
    ASSERT_EQ(cache.size(), 2U);

    cache.evict_one();
    ASSERT_EQ(cache.size(), 1U);

    cache.evict_one();
    ASSERT_EQ(cache.size(), 0U);
}

TEST(moe_lru_cache, evict_on_empty_is_safe) {
    LRUCache cache(4);
    // Should not crash
    cache.evict_one();
    ASSERT_EQ(cache.size(), 0U);
}

// --------------------------------------------------
// Capacity = 1 edge case
// --------------------------------------------------

TEST(moe_lru_cache, capacity_one) {
    LRUCache cache(1);

    auto [s0, h0] = cache.get_lru_item(0);
    ASSERT_EQ(s0, 0U);
    ASSERT_FALSE(h0);

    cache.set_filled(s0);

    // Access same -> hit
    auto [s0b, h0b] = cache.get_lru_item(0);
    ASSERT_EQ(s0b, 0U);
    ASSERT_TRUE(h0b);

    // Insert new -> evicts the only entry, reuses slot 0
    auto [s1, h1] = cache.get_lru_item(1);
    ASSERT_EQ(s1, 0U);
    ASSERT_FALSE(h1);
    ASSERT_EQ(cache.size(), 1U);
}

// --------------------------------------------------
// Stress: many inserts and evictions
// --------------------------------------------------

TEST(moe_lru_cache, stress_many_experts) {
    const size_t cap = 8;
    LRUCache cache(cap);

    // Insert 100 unique (layer=0, expert=i) entries into a cache with capacity 8
    for (size_t i = 0; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(i);
        ASSERT_LT(slot, cap);
        if (i < cap) {
            ASSERT_EQ(slot, i);
        }
    }
    ASSERT_EQ(cache.size(), cap);

    // The last `cap` experts should still be in the cache
    for (size_t i = 100 - cap; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(i);
        // They are in cache (though not filled)
        ASSERT_LT(slot, cap);
    }
}

// --------------------------------------------------
// Thread safety: concurrent get_lru_item calls
// --------------------------------------------------

TEST(moe_lru_cache, concurrent_access) {
    const size_t cap = 16;
    LRUCache cache(cap);
    const int num_threads = 4;
    const int ops_per_thread = 200;

    std::vector<std::thread> threads;
    std::atomic<bool> start{false};

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load()) {}  // spin until all threads are ready
            for (int i = 0; i < ops_per_thread; ++i) {
                size_t expert = t * 32 + i % 32;
                auto [slot, hit] = cache.get_lru_item(expert);
                ASSERT_LT(slot, cap);
                if (!hit) {
                    cache.set_filled(slot);
                }
            }
        });
    }

    start.store(true);
    for (auto& th : threads) {
        th.join();
    }

    ASSERT_LE(cache.size(), cap);
}

