// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"

using namespace cldnn;
using namespace tests;
using ov::intel_gpu::ocl::moe::OffloadExpertWeightProvider;
using ov::intel_gpu::ocl::moe::ResidentExpertWeightProvider;

TEST(moe_expert_weight_provider, resident_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_FALSE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 0U);

    // Fully resident: try_acquire_simultaneous returns the expert ids unchanged, including duplicates
    // (no remap, no dedup) - the expert id is already the addressable slot.
    std::vector<uint32_t> experts = {3, 1, 4, 1, 5, 9, 2, 6};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 1, 4, 1, 5, 9, 2, 6}));
}

TEST(moe_expert_weight_provider, offload_construction_state) {
    MOECompressed::Config config{};
    std::vector<size_t> offsets = {0, 100, 200, 300, 400, 500, 600, 700, 800};
    OffloadExpertWeightProvider provider(/*capacity=*/16, config, offsets, "/path/to/weights.bin");

    ASSERT_TRUE(provider.is_offloaded());
    ASSERT_EQ(provider.resident_capacity(), 16U);
    // Resident buffers are bound lazily at first execute, not at construction.
    ASSERT_FALSE(provider.is_bound());
    ASSERT_FALSE(provider.cache().is_initialized());
}

TEST(moe_expert_weight_provider, offload_bind_resident_buffers) {
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/8, config, {}, "/path/to/weights.bin");
    ASSERT_FALSE(provider.is_bound());

    cldnn::moe_weights resident;  // null device pointers are fine; only the binding flag matters here
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
    ASSERT_TRUE(provider.cache().is_initialized());

    // Idempotent: a second bind keeps the provider bound.
    provider.bind(resident);
    ASSERT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, offload_bind_rejects_insufficient_physical_capacity) {
    auto& engine = get_test_engine();
    MOECompressed::Config config{};
    config.num_expert = 4;
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {}, "/nonexistent");

    const layout full_layout({4, 1, 1, 16}, data_types::u8, format::bfyx);
    const layout one_slot_layout({1, 1, 1, 16}, data_types::u8, format::bfyx);
    auto one_slot_memory = engine.allocate_memory(one_slot_layout, true);
    auto undersized_memory = engine.reinterpret_buffer(*one_slot_memory, full_layout);

    cldnn::moe_weights undersized{};
    undersized.gate_w = undersized_memory;
    EXPECT_THROW(provider.bind(undersized), ov::AssertFailure);
    EXPECT_FALSE(provider.is_bound());

    const layout two_slot_layout({2, 1, 1, 16}, data_types::u8, format::bfyx);
    auto two_slot_memory = engine.allocate_memory(two_slot_layout, true);
    auto sufficient_memory = engine.reinterpret_buffer(*two_slot_memory, full_layout);
    cldnn::moe_weights sufficient{};
    sufficient.gate_w = sufficient_memory;
    EXPECT_NO_THROW(provider.bind(sufficient));
    EXPECT_TRUE(provider.is_bound());
}

TEST(moe_expert_weight_provider, offload_bind_rejects_invalid_logical_sizes) {
    auto& engine = get_test_engine();
    MOECompressed::Config config{};
    config.num_expert = 4;
    const layout backing_layout({1, 1, 1, 16}, data_types::u8, format::bfyx);
    auto backing_memory = engine.allocate_memory(backing_layout, true);

    for (const auto buffer : {&moe_weights::gate_w, &moe_weights::up_w, &moe_weights::down_w, &moe_weights::gate_s, &moe_weights::up_s, &moe_weights::down_s}) {
        for (const size_t bytes_count : {0U, 1U, 2U, 10U}) {
            SCOPED_TRACE(bytes_count);
            OffloadExpertWeightProvider provider(/*capacity=*/2, config, {}, "/nonexistent");
            const layout invalid_layout({1, 1, 1, static_cast<int64_t>(bytes_count)}, data_types::u8, format::bfyx);
            ASSERT_EQ(invalid_layout.bytes_count(), bytes_count);
            cldnn::moe_weights resident{};
            resident.*buffer = engine.reinterpret_buffer(*backing_memory, invalid_layout);

            EXPECT_THROW(provider.bind(resident), ov::AssertFailure);
            EXPECT_FALSE(provider.is_bound());
            EXPECT_FALSE(provider.cache().is_initialized());
        }
    }
}

TEST(moe_expert_weight_provider, offload_bind_accepts_absent_and_scalar_zero_points) {
    auto& engine = get_test_engine();
    const layout scalar_layout({1, 1, 1, 1}, data_types::u8, format::bfyx);
    auto scalar_memory = engine.allocate_memory(scalar_layout, true);
    for (const bool has_zp : {false, true}) {
        SCOPED_TRACE(has_zp);
        MOECompressed::Config config{};
        config.num_expert = 4;
        config.has_zp = has_zp;
        OffloadExpertWeightProvider provider(/*capacity=*/2, config, {}, "/nonexistent");
        const layout zp_layout({1, 1, 1, has_zp ? 1 : 0}, data_types::u8, format::bfyx);
        auto zp_memory = engine.reinterpret_buffer(*scalar_memory, zp_layout);
        cldnn::moe_weights resident{};
        resident.gate_z = zp_memory;
        resident.up_z = zp_memory;
        resident.down_z = zp_memory;

        EXPECT_NO_THROW(provider.bind(resident));
        EXPECT_TRUE(provider.is_bound());
        EXPECT_TRUE(provider.cache().is_initialized());
    }
}

TEST(moe_expert_weight_provider, offload_bind_validates_per_expert_zero_points) {
    auto& engine = get_test_engine();
    MOECompressed::Config config{};
    config.num_expert = 4;
    config.has_zp = true;
    const layout backing_layout({1, 1, 1, 16}, data_types::u8, format::bfyx);
    auto backing_memory = engine.allocate_memory(backing_layout, true);
    for (const auto zero_point : {&moe_weights::gate_z, &moe_weights::up_z, &moe_weights::down_z}) {
        for (const size_t bytes_count : {0U, 2U, 10U, 64U, 32U}) {
            SCOPED_TRACE(bytes_count);
            OffloadExpertWeightProvider provider(/*capacity=*/2, config, {}, "/nonexistent");
            const layout zp_layout({1, 1, 1, static_cast<int64_t>(bytes_count)}, data_types::u8, format::bfyx);
            cldnn::moe_weights resident{};
            resident.*zero_point = engine.reinterpret_buffer(*backing_memory, zp_layout);
            if (bytes_count == 32) {
                EXPECT_NO_THROW(provider.bind(resident));
                EXPECT_TRUE(provider.is_bound());
                EXPECT_TRUE(provider.cache().is_initialized());
            } else {
                EXPECT_THROW(provider.bind(resident), ov::AssertFailure);
                EXPECT_FALSE(provider.is_bound());
                EXPECT_FALSE(provider.cache().is_initialized());
            }
        }
    }
}

TEST(moe_expert_weight_provider, resident_try_acquire_simultaneous_is_identity) {
    ResidentExpertWeightProvider provider;
    std::vector<uint32_t> experts = {3, 7, 3, 1, 7};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    ASSERT_EQ(*lease, (std::vector<size_t>{3, 7, 3, 1, 7}));
}

TEST(moe_expert_weight_provider, resident_acquire_one_is_identity) {
    ResidentExpertWeightProvider provider;
    ASSERT_EQ(provider.acquire_one(42, get_test_stream()), 42U);
    ASSERT_EQ(provider.acquire_one(0, get_test_stream()), static_cast<size_t>(0));
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_capacity_overflow) {
    // Capacity=2 means at most 2 unique experts can be acquired simultaneously.
    // Requesting 3 unique experts must return nullopt (capacity overflow).
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate the LRU cache so experts 0,1 are hits (no disk I/O needed)
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 3 unique experts (0, 1, 2) exceeds capacity=2 -> nullopt
    std::vector<uint32_t> experts = {0, 1, 2, 0, 1};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_FALSE(lease.has_value());
}

TEST(moe_expert_weight_provider, offload_try_acquire_simultaneous_within_capacity) {
    // Same setup but with only 2 unique experts -> should succeed
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    cache.get_lru_item(0);
    cache.set_filled(0);
    cache.get_lru_item(1);
    cache.set_filled(1);

    // 2 unique experts (0, 1) with duplicates -> within capacity=2
    std::vector<uint32_t> experts = {0, 1, 0, 1, 0};
    auto lease = provider.try_acquire_simultaneous(experts, get_test_stream());
    ASSERT_TRUE(lease.has_value());
    // Slots should be valid (cache returned slots for experts 0 and 1)
    ASSERT_EQ(lease->size(), 5U);
    // Duplicates must map to the same slot
    ASSERT_EQ((*lease)[0], (*lease)[2]);
    ASSERT_EQ((*lease)[0], (*lease)[4]);
    ASSERT_EQ((*lease)[1], (*lease)[3]);
}

TEST(moe_expert_weight_provider, offload_acquire_one_basic) {
    // acquire_one always succeeds (evicts LRU if needed)
    MOECompressed::Config config{};
    OffloadExpertWeightProvider provider(/*capacity=*/2, config, {0, 100, 200, 300}, "/nonexistent");

    cldnn::moe_weights resident{};
    provider.bind(resident);

    // Pre-populate cache
    auto& cache = provider.cache();
    auto [slot0, hit0] = cache.get_lru_item(5);
    cache.set_filled(slot0);

    // acquire_one should return a valid slot
    auto slot = provider.acquire_one(5, get_test_stream());
    ASSERT_EQ(slot, slot0);
}
