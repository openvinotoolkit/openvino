// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/test_utils.h"
#include "ocl_v2/moe/expert_weight_providers.hpp"

#include <filesystem>
#include <fstream>

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
    std::vector<size_t> offsets = {0, 100, 200};
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

// The weights are streamed from the weights file, the scales and zero points are copied from their resident tensors
TEST(moe_expert_weight_provider, offload_loads_weights_from_file_and_params_from_resident_tensors) {
    auto& engine = get_test_engine();
    auto& stream = get_test_stream();
    constexpr size_t num_expert = 4, capacity = 2, rows = 8, cols = 16, groups = 2;
    constexpr size_t weights_offset = 64, weight_bytes = rows * cols;

    std::vector<uint8_t> file_data(weights_offset + num_expert * weight_bytes, 0xFF);
    for (size_t e = 0; e < num_expert; e++) {
        std::fill_n(file_data.begin() + weights_offset + e * weight_bytes, weight_bytes, static_cast<uint8_t>(e));
    }
    const auto path = std::filesystem::temp_directory_path() / "moe_expert_weight_provider_test.bin";
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(file_data.data()), static_cast<std::streamsize>(file_data.size()));
    }

    const layout weight_layout{ov::Shape{num_expert, rows, cols}, data_types::u8, format::bfyx};
    const layout weight_slots_layout{ov::Shape{capacity, rows, cols}, data_types::u8, format::bfyx};
    auto weight_slots = engine.reinterpret_buffer(*engine.allocate_memory(weight_slots_layout), weight_layout);
    // Scales already reordered by the regular pipeline, with every expert resident
    const layout scales_layout{ov::Shape{num_expert, rows, groups, 1}, data_types::f16, format::byfx};
    auto scales = engine.allocate_memory(scales_layout);
    std::vector<ov::float16> scale_data(num_expert * rows * groups);
    for (size_t i = 0; i < scale_data.size(); i++) {
        scale_data[i] = ov::float16(static_cast<float>(i));
    }
    set_values(scales, scale_data);
    const layout shared_zp_layout{ov::Shape{1}, data_types::u8, format::bfyx};
    auto shared_zp = engine.allocate_memory(shared_zp_layout);

    MOECompressed::Config config{};
    config.num_expert = num_expert;
    {
        OffloadExpertWeightProvider provider(capacity, config, {weights_offset, 0, 0}, path);
        cldnn::moe_weights resident{};
        resident.gate_w = weight_slots;
        resident.gate_s = scales;
        resident.gate_z = shared_zp;
        provider.bind(resident);
        ASSERT_NE(resident.gate_s, scales);
        ASSERT_EQ(resident.gate_z, shared_zp);

        const uint32_t expert = 3;
        const auto slot = provider.acquire_one(expert, stream);
        ASSERT_LT(slot, capacity);

        std::vector<uint8_t> loaded_weights(weight_bytes);
        weight_slots->copy_to(stream, loaded_weights.data(), slot * weight_bytes, 0, weight_bytes, true);
        EXPECT_EQ(loaded_weights, std::vector<uint8_t>(weight_bytes, static_cast<uint8_t>(expert)));

        constexpr size_t scale_count = rows * groups;
        std::vector<ov::float16> loaded_scales(scale_count);
        resident.gate_s->copy_to(stream, loaded_scales.data(), slot * scale_count * sizeof(ov::float16), 0, scale_count * sizeof(ov::float16), true);
        EXPECT_EQ(loaded_scales, std::vector<ov::float16>(scale_data.begin() + expert * scale_count, scale_data.begin() + (expert + 1) * scale_count));
    }
    std::filesystem::remove(path);
}
