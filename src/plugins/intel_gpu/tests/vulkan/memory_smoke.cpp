// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "gtest/gtest.h"
#include "intel_gpu/runtime/device_query.hpp"
#include "vulkan/vulkan_device.hpp"
#include "vulkan/vulkan_engine.hpp"
#include "vulkan/vulkan_memory.hpp"
#include "vulkan/vulkan_tensor_shape.hpp"

namespace cldnn::vulkan::test {

class vulkan_buffer_allocation_access {
public:
    static void force_staging(vulkan_buffer_allocation& allocation) {
        allocation.unmap();
    }

    static void force_non_coherent(vulkan_buffer_allocation& allocation, VkDeviceSize atom_size) {
        allocation._memory_flags &= ~static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        allocation._non_coherent_atom_size = atom_size;
    }
};

}  // namespace cldnn::vulkan::test

namespace {

using cldnn::vulkan::test::vulkan_buffer_allocation_access;

cldnn::layout byte_layout(int64_t count) {
    return {ov::PartialShape{1, 1, 1, count}, ov::element::u8, cldnn::format::bfyx};
}

void check_disjoint_locks(cldnn::engine& engine, cldnn::memory& buffer, cldnn::stream& stream, const uint8_t* mapped_base = nullptr) {
    auto first = engine.create_subbuffer(buffer, byte_layout(2), 1);
    auto second = engine.create_subbuffer(buffer, byte_layout(2), 3);
    auto* first_data = static_cast<uint8_t*>(first->lock(stream, cldnn::mem_lock_type::write));
    first_data[0] = 11;
    first_data[1] = 12;
    auto* second_data = static_cast<uint8_t*>(second->lock(stream, cldnn::mem_lock_type::read_write));
    second_data[0] = 21;
    second_data[1] = 22;
    first->unlock(stream);
    second->unlock(stream);

    std::array<uint8_t, 8> actual{};
    buffer.copy_to(stream, actual.data(), 0, 0, actual.size(), true);
    constexpr std::array<uint8_t, 8> expected{0, 11, 12, 21, 22, 0, 0, 0};
    EXPECT_EQ(actual, expected) << "Unlocking a staged subbuffer lost a write to a disjoint view";
    if (mapped_base != nullptr) {
        EXPECT_NE(first_data, mapped_base + 1);
        EXPECT_NE(second_data, mapped_base + 3);
    }
}

void check_mapped_non_coherent_locks(cldnn::vulkan::vulkan_engine& engine, cldnn::stream& stream) {
    using namespace cldnn::vulkan;
    const auto atom_size = std::max<VkDeviceSize>(engine.get_non_coherent_atom_size(), 8);
    auto allocator = std::make_shared<vulkan_memory_allocator>(atom_size, atom_size, atom_size, [&](size_t size) {
        return vulkan_buffer_allocation::create(engine, size, vulkan_buffer_memory_usage::host_staging);
    });
    auto region = allocator->allocate(8);
    const auto& allocation = region->get_allocation();
    const auto* mapped = static_cast<const uint8_t*>(allocation->get_mapped_data());
    ASSERT_NE(mapped, nullptr) << "The non-coherent test requires an active mapping";
    // Exercise the non-coherent policy on devices with coherent mappings or single-byte atoms.
    vulkan_buffer_allocation_access::force_non_coherent(*allocation, atom_size);
    vulkan_buffer buffer(&engine, byte_layout(static_cast<int64_t>(atom_size)), region, 0, nullptr);
    buffer.fill(stream);
    check_disjoint_locks(engine, buffer, stream, mapped);
    const auto* whole_atom = buffer.lock(stream, cldnn::mem_lock_type::read);
    buffer.unlock(stream);
    EXPECT_EQ(whole_atom, mapped) << "An entire non-coherent atom must retain direct mapping access";
}

void check_padded_allocation_fallbacks(cldnn::vulkan::vulkan_engine& engine, cldnn::stream& stream) {
    using namespace cldnn::vulkan;
    constexpr size_t requested = 257;
    constexpr size_t padded = 260;
    constexpr size_t preferred = 1024;
    const auto check_fallback = [&](size_t limit, bool fail_pooled_allocation) {
        std::vector<size_t> attempted;
        auto allocator = std::make_shared<vulkan_memory_allocator>(sizeof(uint32_t), preferred, limit, [&](size_t size) {
            attempted.push_back(size);
            if (fail_pooled_allocation && attempted.size() == 1) {
                throw std::bad_alloc{};
            }
            return vulkan_buffer_allocation::create(engine, size, vulkan_buffer_memory_usage::host_staging);
        });
        auto region = allocator->allocate(requested);
        ASSERT_EQ(attempted.size(), fail_pooled_allocation ? 2U : 1U);
        EXPECT_EQ(attempted.back(), padded);
        ASSERT_EQ(region->get_size(), padded) << "Allocation fallback discarded transfer padding";
        vulkan_buffer_allocation_access::force_staging(*region->get_allocation());
        vulkan_buffer buffer(&engine, byte_layout(requested), region, 0, nullptr);
        constexpr uint8_t pattern = 0x5A;
        buffer.fill(stream, pattern);
        std::array<uint8_t, requested> output{};
        buffer.copy_to(stream, output.data(), 0, 0, output.size(), true);
        std::array<uint8_t, requested> expected;
        expected.fill(pattern);
        EXPECT_EQ(output, expected) << "The padded fallback did not preserve every staged byte";
    };
    check_fallback(requested, false);
    check_fallback(preferred, true);

    std::vector<size_t> attempted;
    vulkan_memory_allocator allocator(sizeof(uint32_t), padded, preferred, [&](size_t size) -> vulkan_buffer_allocation::ptr {
        attempted.push_back(size);
        throw std::bad_alloc{};
    });
    EXPECT_THROW(allocator.allocate(requested), std::bad_alloc);
    EXPECT_EQ(attempted, std::vector<size_t>{padded}) << "An already padded request must not retry";
}

void check_strided_rows(cldnn::memory& buffer, cldnn::stream& stream) {
    constexpr size_t rows = 3;
    constexpr size_t columns = 6;
    constexpr size_t stride = 16;
    constexpr size_t buffer_offset = 61;
    std::array<uint8_t, rows * stride> input{};
    std::array<uint8_t, rows * stride> output{};
    output.fill(0xA5);
    auto expected = output;
    for (size_t row = 0; row < rows; ++row) {
        const auto host_offset = row * stride;
        const auto device_offset = buffer_offset + row * columns;
        for (size_t col = 0; col < columns; ++col) {
            input[host_offset + col] = static_cast<uint8_t>(1 + row * columns + col);
            expected[host_offset + col] = input[host_offset + col];
        }
        buffer.copy_from(stream, input.data(), host_offset, device_offset, columns, true);
    }
    for (size_t row = 0; row < rows; ++row) {
        buffer.copy_to(stream, output.data(), buffer_offset + row * columns, row * stride, columns, true);
    }
    EXPECT_EQ(output, expected) << "Staged strided transfer changed payload or padding";
    std::array<uint8_t, 2> boundaries{1, 1};
    buffer.copy_to(stream, boundaries.data(), buffer_offset - 1, 0, 1, true);
    buffer.copy_to(stream, boundaries.data(), buffer_offset + rows * columns, 1, 1, true);
    EXPECT_EQ(boundaries, (std::array<uint8_t, 2>{})) << "Staged upload changed adjacent bytes";
}

void check_device_copies(cldnn::engine& engine, cldnn::stream& stream) {
    using cldnn::vulkan::vulkan_buffer;
    for (const size_t count : {256, 257}) {
        auto source = engine.allocate_memory(byte_layout(count + 4), cldnn::allocation_type::vk_buffer, false);
        auto destination = engine.allocate_memory(byte_layout(count + 4), cldnn::allocation_type::vk_buffer, false);
        vulkan_buffer_allocation_access::force_staging(*static_cast<vulkan_buffer&>(*source).get_allocation());
        vulkan_buffer_allocation_access::force_staging(*static_cast<vulkan_buffer&>(*destination).get_allocation());
        destination->fill(stream, 0xA5);

        std::vector<uint8_t> expected(count + 4, 0xA5);
        for (size_t i = 0; i < count; ++i) {
            expected[i + 2] = static_cast<uint8_t>(i);
        }
        source->copy_from(stream, expected.data() + 2, 0, 1, count, true);
        source->copy_to(stream, *destination, 1, 2, count, true);
        std::vector<uint8_t> result(expected.size());
        destination->copy_to(stream, result.data(), 0, 0, result.size(), true);
        EXPECT_EQ(result, expected) << "Device copy changed the payload or adjacent bytes";
    }
}

TEST(VulkanMemorySmoke, ValidatesTensorSizeArithmetic) {
    using cldnn::vulkan::validate_tensor_shape;
    EXPECT_NO_THROW(validate_tensor_shape({}, ov::element::u8));
    EXPECT_NO_THROW(validate_tensor_shape({3, 0, 5}, ov::element::f16));
    EXPECT_NO_THROW(validate_tensor_shape({3}, ov::element::u4));

    const auto large_extent = (std::numeric_limits<size_t>::max)() / 4;
    EXPECT_THROW(validate_tensor_shape({large_extent, 5}, ov::element::u8), ov::Exception);
    EXPECT_THROW(validate_tensor_shape({large_extent}, ov::element::u8), ov::Exception);
}

TEST(VulkanMemorySmoke, ReportsAllocationLimitWithinDeviceLimits) {
    const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
    ASSERT_FALSE(devices.empty()) << "No Vulkan GPU detected";
    for (const auto& [id, device] : devices) {
        SCOPED_TRACE(id);
        const auto physical_device = static_cast<const cldnn::vulkan::vulkan_device&>(*device).get_physical_device();
        VkPhysicalDeviceVulkan13Properties properties13{};
        properties13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
        VkPhysicalDeviceVulkan11Properties properties11{};
        properties11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
        properties11.pNext = &properties13;
        VkPhysicalDeviceProperties2 properties{};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties.pNext = &properties11;
        vkGetPhysicalDeviceProperties2(physical_device, &properties);

        const auto& info = device->get_info();
        EXPECT_GT(info.max_alloc_mem_size, 0U);
        EXPECT_LE(info.max_alloc_mem_size, info.max_global_mem_size);
        EXPECT_LE(info.max_alloc_mem_size, properties.properties.limits.maxStorageBufferRange);
        EXPECT_LE(info.max_alloc_mem_size, properties11.maxMemoryAllocationSize);
        EXPECT_LE(info.max_alloc_mem_size, properties13.maxBufferSize);
    }
}

VkDeviceSize get_backing_memory_size(const cldnn::vulkan::vulkan_buffer_allocation& allocation) {
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(allocation.get_device(), allocation.get_buffer(), &requirements);
    return requirements.size;
}

TEST(VulkanMemorySmoke, CountsStagingAndRetainedPoolMemory) {
    using namespace cldnn::vulkan;
    const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
    ASSERT_FALSE(devices.empty()) << "No Vulkan GPU detected";
    const auto device = std::static_pointer_cast<vulkan_device>(devices.begin()->second);
    const auto can_reserve_one_byte = [&](VkDeviceSize limit) {
        if (!device->try_reserve_memory(1, limit)) {
            return false;
        }
        device->release_memory(1);
        return true;
    };

    {
        auto engine = cldnn::engine::create(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan, device);
        auto buffer = engine->allocate_memory(byte_layout(64), cldnn::allocation_type::vk_buffer, false);
        const auto& backing = *static_cast<vulkan_buffer&>(*buffer).get_allocation();
        const auto backing_size = get_backing_memory_size(backing);
        const auto backing_handle = backing.get_buffer();
        EXPECT_FALSE(can_reserve_one_byte(backing_size));
        EXPECT_TRUE(can_reserve_one_byte(backing_size + 1));

        {
            auto& vk_engine = static_cast<vulkan_engine&>(*engine);
            auto staging = vulkan_buffer_allocation::create(vk_engine, 32, vulkan_buffer_memory_usage::host_staging);
            const auto combined_size = backing_size + get_backing_memory_size(*staging);
            EXPECT_FALSE(can_reserve_one_byte(combined_size));
            EXPECT_TRUE(can_reserve_one_byte(combined_size + 1));
        }
        EXPECT_TRUE(can_reserve_one_byte(backing_size + 1));

        buffer.reset();
        EXPECT_EQ(engine->get_used_device_memory(cldnn::allocation_type::vk_buffer), 0U);
        EXPECT_FALSE(can_reserve_one_byte(backing_size));
        auto reused = engine->allocate_memory(byte_layout(64), cldnn::allocation_type::vk_buffer, false);
        EXPECT_EQ(static_cast<vulkan_buffer&>(*reused).get_allocation()->get_buffer(), backing_handle);
        EXPECT_TRUE(can_reserve_one_byte(backing_size + 1));
    }

    EXPECT_TRUE(can_reserve_one_byte(1));
}

TEST(VulkanMemorySmoke, RejectsPhysicalAllocationsOnBudgetExhaustion) {
    using namespace cldnn::vulkan;
    const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
    ASSERT_FALSE(devices.empty()) << "No Vulkan GPU detected";
    auto engine = cldnn::engine::create(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan, devices.begin()->second);
    auto& vk_engine = static_cast<vulkan_engine&>(*engine);
    const auto device = vk_engine.get_vulkan_device_object();

    // Simulate a full budget without allocating anything to fill it.
    constexpr auto reserved = (std::numeric_limits<VkDeviceSize>::max)();
    ASSERT_TRUE(device->try_reserve_memory(reserved, reserved));
    EXPECT_THROW(vulkan_buffer_allocation::create(vk_engine, 64, vulkan_buffer_memory_usage::device), std::bad_alloc);
    EXPECT_THROW(vulkan_buffer_allocation::create(vk_engine, 32, vulkan_buffer_memory_usage::host_staging), std::bad_alloc);
    device->release_memory(reserved);
    EXPECT_NO_THROW(vulkan_buffer_allocation::create(vk_engine, 32, vulkan_buffer_memory_usage::host_staging));
}

TEST(VulkanMemorySmoke, ChecksStagedTransfersAndAllocations) {
    const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
    ASSERT_FALSE(devices.empty()) << "No Vulkan GPU detected";
    auto engine = cldnn::engine::create(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan, devices.begin()->second);
    auto buffer = engine->allocate_memory(byte_layout(4096), cldnn::allocation_type::vk_buffer, false);
    vulkan_buffer_allocation_access::force_staging(*static_cast<cldnn::vulkan::vulkan_buffer&>(*buffer).get_allocation());
    auto& stream = engine->get_service_stream();
    buffer->fill(stream);
    check_disjoint_locks(*engine, *buffer, stream);
    check_strided_rows(*buffer, stream);
    check_device_copies(*engine, stream);
    auto& vulkan_engine = static_cast<cldnn::vulkan::vulkan_engine&>(*engine);
    check_mapped_non_coherent_locks(vulkan_engine, stream);
    check_padded_allocation_fallbacks(vulkan_engine, stream);
}

}  // namespace
