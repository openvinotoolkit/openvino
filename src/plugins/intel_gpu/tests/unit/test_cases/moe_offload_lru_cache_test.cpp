// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "ocl_v2/moe/LRUCache.hpp"

#include <thread>
#include <vector>
#include <set>
#include <atomic>

using namespace cldnn;
using namespace tests;

// ──────────────────────────────────────────────────
// Basic construction and initial state
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, initial_state) {
    LRUCache cache(4);

    ASSERT_EQ(cache.size(), 0U);
    ASSERT_EQ(cache.get_total_experts(), 0U);
    ASSERT_FALSE(cache.m_initialized);
}

// ──────────────────────────────────────────────────
// get_lru_item: insert (miss) and hit
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, single_insert_is_miss) {
    LRUCache cache(4);

    auto [slot, hit] = cache.get_lru_item(/*layer=*/0, /*expert=*/0);
    ASSERT_FALSE(hit);        // first access is a miss
    ASSERT_EQ(slot, 0U);     // first slot assigned
    ASSERT_EQ(cache.size(), 1U);
}

TEST(moe_lru_cache, second_access_without_fill_is_miss) {
    // get_lru_item returns is_hit based on m_filled_list, not just presence
    LRUCache cache(4);

    cache.get_lru_item(0, 0);  // insert, slot 0
    auto [slot, hit] = cache.get_lru_item(0, 0);  // access again

    ASSERT_EQ(slot, 0U);
    ASSERT_FALSE(hit);  // filled_list[0] is still false
}

TEST(moe_lru_cache, access_after_set_filled_is_hit) {
    LRUCache cache(4);

    auto [slot, miss] = cache.get_lru_item(0, 0);
    cache.set_filled(slot);

    auto [slot2, hit] = cache.get_lru_item(0, 0);
    ASSERT_EQ(slot2, slot);
    ASSERT_TRUE(hit);
}

// ──────────────────────────────────────────────────
// Sequential slot allocation
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, slots_assigned_sequentially) {
    const size_t cap = 4;
    LRUCache cache(cap);

    for (size_t i = 0; i < cap; ++i) {
        auto [slot, hit] = cache.get_lru_item(/*layer=*/0, /*expert=*/i);
        ASSERT_EQ(slot, i);
        ASSERT_FALSE(hit);
    }
    ASSERT_EQ(cache.size(), cap);
}

// ──────────────────────────────────────────────────
// Eviction: oldest entry evicted when cache is full
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, eviction_when_full) {
    LRUCache cache(3);

    // Fill cache: experts 0, 1, 2
    cache.get_lru_item(0, 0);  // slot 0
    cache.get_lru_item(0, 1);  // slot 1
    cache.get_lru_item(0, 2);  // slot 2
    ASSERT_EQ(cache.size(), 3U);

    // Insert expert 3 → evicts expert 0 (oldest), reuses slot 0
    auto [slot, hit] = cache.get_lru_item(0, 3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 0U);   // expert 0's slot is recycled
    ASSERT_EQ(cache.size(), 3U);  // size stays at capacity
}

TEST(moe_lru_cache, evicted_entry_becomes_miss) {
    LRUCache cache(2);

    // Fill: expert 0 (slot 0), expert 1 (slot 1)
    cache.get_lru_item(0, 0);
    cache.set_filled(0);
    cache.get_lru_item(0, 1);
    cache.set_filled(1);

    // Insert expert 2 → evicts expert 0
    cache.get_lru_item(0, 2);

    // Access expert 0 again → should be a miss (it was evicted)
    auto [slot, hit] = cache.get_lru_item(0, 0);
    ASSERT_FALSE(hit);
}

// ──────────────────────────────────────────────────
// LRU ordering: recently accessed items survive eviction
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, lru_order_refresh_on_access) {
    LRUCache cache(3);

    // Insert experts 0, 1, 2 in order
    cache.get_lru_item(0, 0);  // slot 0  (LRU order: 0)
    cache.get_lru_item(0, 1);  // slot 1  (LRU order: 0, 1)
    cache.get_lru_item(0, 2);  // slot 2  (LRU order: 0, 1, 2)

    // Access expert 0 → moves to most recent
    // (LRU order: 1, 2, 0)
    cache.get_lru_item(0, 0);

    // Insert expert 3 → should evict expert 1 (now the oldest)
    auto [slot, hit] = cache.get_lru_item(0, 3);
    ASSERT_FALSE(hit);
    ASSERT_EQ(slot, 1U);  // reuses expert 1's slot
}

TEST(moe_lru_cache, double_refresh_changes_eviction_order) {
    LRUCache cache(3);

    // Insert 0, 1, 2
    cache.get_lru_item(0, 0);  // slot 0
    cache.get_lru_item(0, 1);  // slot 1
    cache.get_lru_item(0, 2);  // slot 2

    // Refresh 0, then refresh 1  → LRU order: 2, 0, 1
    cache.get_lru_item(0, 0);
    cache.get_lru_item(0, 1);

    // Insert 3 → evicts expert 2 (oldest)
    auto [slot3, hit3] = cache.get_lru_item(0, 3);
    ASSERT_FALSE(hit3);
    ASSERT_EQ(slot3, 2U);  // reuses expert 2's slot

    // Insert 4 → evicts expert 0 (now oldest)
    auto [slot4, hit4] = cache.get_lru_item(0, 4);
    ASSERT_FALSE(hit4);
    ASSERT_EQ(slot4, 0U);  // reuses expert 0's slot
}

// ──────────────────────────────────────────────────
// Multi-layer: (layer, expert) pairs are independent
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, multi_layer_keys_are_independent) {
    LRUCache cache(4);

    auto [s0, h0] = cache.get_lru_item(/*layer=*/0, /*expert=*/0);
    auto [s1, h1] = cache.get_lru_item(/*layer=*/1, /*expert=*/0);

    ASSERT_FALSE(h0);
    ASSERT_FALSE(h1);
    ASSERT_NE(s0, s1);  // different slots for different layers
    ASSERT_EQ(cache.size(), 2U);
}

TEST(moe_lru_cache, same_expert_different_layers_both_cached) {
    LRUCache cache(4);

    cache.get_lru_item(0, 5);
    cache.set_filled(0);
    cache.get_lru_item(1, 5);
    cache.set_filled(1);

    auto [s0, h0] = cache.get_lru_item(0, 5);
    auto [s1, h1] = cache.get_lru_item(1, 5);

    ASSERT_TRUE(h0);
    ASSERT_TRUE(h1);
    ASSERT_NE(s0, s1);
}

// ──────────────────────────────────────────────────
// set_filled / filled tracking
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, filled_cleared_on_eviction) {
    LRUCache cache(2);

    // Insert expert 0, mark filled
    auto [s0, miss0] = cache.get_lru_item(0, 0);
    cache.set_filled(s0);

    // Insert expert 1
    cache.get_lru_item(0, 1);

    // Insert expert 2 → evicts expert 0, reuses slot s0
    auto [s2, h2] = cache.get_lru_item(0, 2);
    ASSERT_EQ(s2, s0);         // recycled slot
    ASSERT_FALSE(h2);          // filled was cleared during eviction

    // Even after re-inserting to the same slot, accessing it returns not-filled
    auto [s2b, h2b] = cache.get_lru_item(0, 2);
    ASSERT_FALSE(h2b);  // not filled until set_filled is called
}

TEST(moe_lru_cache, set_filled_out_of_range_is_safe) {
    LRUCache cache(2);
    // Should not crash
    cache.set_filled(100);
    cache.set_filled(std::numeric_limits<size_t>::max());
}

// ──────────────────────────────────────────────────
// evict_one: explicit eviction
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, explicit_evict_reduces_size) {
    LRUCache cache(4);

    cache.get_lru_item(0, 0);
    cache.get_lru_item(0, 1);
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

// ──────────────────────────────────────────────────
// Capacity = 1 edge case
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, capacity_one) {
    LRUCache cache(1);

    auto [s0, h0] = cache.get_lru_item(0, 0);
    ASSERT_EQ(s0, 0U);
    ASSERT_FALSE(h0);

    cache.set_filled(s0);

    // Access same → hit
    auto [s0b, h0b] = cache.get_lru_item(0, 0);
    ASSERT_EQ(s0b, 0U);
    ASSERT_TRUE(h0b);

    // Insert new → evicts the only entry, reuses slot 0
    auto [s1, h1] = cache.get_lru_item(0, 1);
    ASSERT_EQ(s1, 0U);
    ASSERT_FALSE(h1);
    ASSERT_EQ(cache.size(), 1U);
}

// ──────────────────────────────────────────────────
// Stress: many inserts and evictions
// ──────────────────────────────────────────────────

TEST(moe_lru_cache, stress_many_experts) {
    const size_t cap = 8;
    LRUCache cache(cap);

    // Insert 100 unique (layer=0, expert=i) entries into a cache with capacity 8
    for (size_t i = 0; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(0, i);
        ASSERT_LT(slot, cap);
        if (i < cap) {
            ASSERT_EQ(slot, i);
        }
    }
    ASSERT_EQ(cache.size(), cap);

    // The last `cap` experts should still be in the cache
    for (size_t i = 100 - cap; i < 100; ++i) {
        auto [slot, hit] = cache.get_lru_item(0, i);
        // They are in cache (though not filled)
        ASSERT_LT(slot, cap);
    }
}

// ──────────────────────────────────────────────────
// Thread safety: concurrent get_lru_item calls
// ──────────────────────────────────────────────────

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
                size_t layer = t;
                size_t expert = i % 32;
                auto [slot, hit] = cache.get_lru_item(layer, expert);
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

// ──────────────────────────────────────────────────
// Property config: moe_offload_max_experts roundtrip
// ──────────────────────────────────────────────────

TEST(moe_offload_property_test, default_value_is_zero) {
    auto config = get_test_default_config(get_test_engine());
    ASSERT_EQ(config.get_moe_offload_max_experts(), 0U);
}

TEST(moe_offload_property_test, set_and_get_various_values) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::moe_offload_max_experts(1));
    ASSERT_EQ(config.get_moe_offload_max_experts(), 1U);

    config.set_property(ov::intel_gpu::moe_offload_max_experts(64));
    ASSERT_EQ(config.get_moe_offload_max_experts(), 64U);

    config.set_property(ov::intel_gpu::moe_offload_max_experts(256));
    ASSERT_EQ(config.get_moe_offload_max_experts(), 256U);
}

TEST(moe_offload_property_test, set_back_to_zero_disables) {
    auto config = get_test_default_config(get_test_engine());

    config.set_property(ov::intel_gpu::moe_offload_max_experts(48));
    ASSERT_EQ(config.get_moe_offload_max_experts(), 48U);

    config.set_property(ov::intel_gpu::moe_offload_max_experts(0));
    ASSERT_EQ(config.get_moe_offload_max_experts(), 0U);
}

// ──────────────────────────────────────────────────
// Primitive: moe_3gemm_fused_compressed fields
// ──────────────────────────────────────────────────

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"

TEST(moe_offload_primitive_test, default_offload_fields) {
    cldnn::moe_3gemm_fused_compressed prim;
    ASSERT_TRUE(prim._weight_bin_offsets.empty());
    ASSERT_TRUE(prim._weights_path.empty());
    ASSERT_EQ(prim._lru_expert_num, 0U);
}

TEST(moe_offload_primitive_test, construct_with_offload_params) {
    MOE3GemmFusedCompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    std::string path = "/path/to/weights.bin";
    size_t lru_num = 16;

    cldnn::moe_3gemm_fused_compressed prim(
        "test_moe",
        {cldnn::input_info("input0"), cldnn::input_info("input1")},
        config,
        offsets,
        path,
        lru_num);

    ASSERT_EQ(prim._weight_bin_offsets.size(), 9U);
    ASSERT_EQ(prim._weight_bin_offsets[0], 0U);
    ASSERT_EQ(prim._weight_bin_offsets[8], 800U);
    ASSERT_EQ(prim._weights_path, path);
    ASSERT_EQ(prim._lru_expert_num, lru_num);
}

TEST(moe_offload_primitive_test, equality_with_offload_fields) {
    MOE3GemmFusedCompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};

    cldnn::moe_3gemm_fused_compressed prim1(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        16);

    cldnn::moe_3gemm_fused_compressed prim2(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        16);

    ASSERT_TRUE(prim1 == prim2);

    // Different weights_path → not equal
    cldnn::moe_3gemm_fused_compressed prim3(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/b.bin",
        16);
    ASSERT_FALSE(prim1 == prim3);

    // Different lru_expert_num → not equal
    cldnn::moe_3gemm_fused_compressed prim4(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets,
        "/path/a.bin",
        32);
    ASSERT_FALSE(prim1 == prim4);

    // Different offsets → not equal
    std::vector<size_t> offsets2 = {0, 100, 200, 300, 400, 500, 600, 700, 999};
    cldnn::moe_3gemm_fused_compressed prim5(
        "test_moe",
        {cldnn::input_info("input0")},
        config,
        offsets2,
        "/path/a.bin",
        16);
    ASSERT_FALSE(prim1 == prim5);
}
