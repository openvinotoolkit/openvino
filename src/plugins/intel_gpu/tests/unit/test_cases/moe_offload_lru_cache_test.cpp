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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Basic construction and initial state
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TEST(moe_lru_cache, initial_state) {
    LRUCache cache(4);

    ASSERT_EQ(cache.size(), 0U);
    ASSERT_FALSE(cache.is_initialized());
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// get_lru_item: insert (miss) and hit
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Sequential slot allocation
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Eviction: oldest entry evicted when cache is full
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TEST(moe_lru_cache, eviction_when_full) {
    LRUCache cache(3);

    // Fill cache: experts 0, 1, 2
    cache.get_lru_item(0);  // slot 0
    cache.get_lru_item(1);  // slot 1
    cache.get_lru_item(2);  // slot 2
    ASSERT_EQ(cache.size(), 3U);

    // Insert expert 3 â†’ evicts expert 0 (oldest), reuses slot 0
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

    // Insert expert 2 â†’ evicts expert 0
    cache.get_lru_item(2);

    // Access expert 0 again â†’ should be a miss (it was evicted)
    auto [slot, hit] = cache.get_lru_item(0);
    ASSERT_FALSE(hit);
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// LRU ordering: recently accessed items survive eviction
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TEST(moe_lru_cache, lru_order_refresh_on_access) {
    LRUCache cache(3);

    // Insert experts 0, 1, 2 in order
    cache.get_lru_item(0);  // slot 0  (LRU order: 0)
    cache.get_lru_item(1);  // slot 1  (LRU order: 0, 1)
    cache.get_lru_item(2);  // slot 2  (LRU order: 0, 1, 2)

    // Access expert 0 â†’ moves to most recent
    // (LRU order: 1, 2, 0)
    cache.get_lru_item(0);

    // Insert expert 3 â†’ should evict expert 1 (now the oldest)
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

    // Refresh 0, then refresh 1  â†’ LRU order: 2, 0, 1
    cache.get_lru_item(0);
    cache.get_lru_item(1);

    // Insert 3 â†’ evicts expert 2 (oldest)
    auto [slot3, hit3] = cache.get_lru_item(3);
    ASSERT_FALSE(hit3);
    ASSERT_EQ(slot3, 2U);  // reuses expert 2's slot

    // Insert 4 â†’ evicts expert 0 (now oldest)
    auto [slot4, hit4] = cache.get_lru_item(4);
    ASSERT_FALSE(hit4);
    ASSERT_EQ(slot4, 0U);  // reuses expert 0's slot
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Multi-layer: (layer, expert) pairs are independent
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// set_filled / filled tracking
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TEST(moe_lru_cache, filled_cleared_on_eviction) {
    LRUCache cache(2);

    // Insert expert 0, mark filled
    auto [s0, miss0] = cache.get_lru_item(0);
    cache.set_filled(s0);

    // Insert expert 1
    cache.get_lru_item(1);

    // Insert expert 2 â†’ evicts expert 0, reuses slot s0
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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// evict_one: explicit eviction
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Capacity = 1 edge case
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TEST(moe_lru_cache, capacity_one) {
    LRUCache cache(1);

    auto [s0, h0] = cache.get_lru_item(0);
    ASSERT_EQ(s0, 0U);
    ASSERT_FALSE(h0);

    cache.set_filled(s0);

    // Access same â†’ hit
    auto [s0b, h0b] = cache.get_lru_item(0);
    ASSERT_EQ(s0b, 0U);
    ASSERT_TRUE(h0b);

    // Insert new â†’ evicts the only entry, reuses slot 0
    auto [s1, h1] = cache.get_lru_item(1);
    ASSERT_EQ(s1, 0U);
    ASSERT_FALSE(h1);
    ASSERT_EQ(cache.size(), 1U);
}

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Stress: many inserts and evictions
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// Thread safety: concurrent get_lru_item calls
// â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
