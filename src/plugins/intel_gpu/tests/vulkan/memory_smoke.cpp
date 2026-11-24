// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <vector>

#include "intel_gpu/runtime/device_query.hpp"
#include "openvino/core/except.hpp"
#include "vulkan/vulkan_engine.hpp"
#include "vulkan/vulkan_memory.hpp"
#include "vulkan/vulkan_memory_internal.hpp"

namespace {

cldnn::layout byte_layout(int64_t count) {
    return {ov::PartialShape{1, 1, 1, count}, ov::element::u8, cldnn::format::bfyx};
}

void unmap_for_staging(const cldnn::vulkan::vulkan_buffer_allocation::ptr& allocation) {
    if (allocation->is_mapped()) {
        // Exercise real staging copies on unified-memory GPUs.
        vkUnmapMemory(allocation->device, allocation->memory);
        allocation->mapped_data = nullptr;
    }
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
    OPENVINO_ASSERT(actual == expected, "Unlocking a staged subbuffer lost a write to a disjoint view");
    OPENVINO_ASSERT(mapped_base == nullptr || (first_data != mapped_base + 1 && second_data != mapped_base + 3),
                    "Partial non-coherent atoms must use staging while the backing allocation stays mapped");
}

void check_mapped_non_coherent_locks(cldnn::vulkan::vulkan_engine& engine, cldnn::stream& stream) {
    using namespace cldnn::vulkan;
    const auto atom_size = std::max<VkDeviceSize>(engine.get_non_coherent_atom_size(), 8);
    auto allocator = std::make_shared<vulkan_memory_allocator>(atom_size, atom_size, atom_size, [&](size_t size) {
        return detail::allocate_vulkan_buffer(engine, size, vulkan_buffer_memory_usage::host_staging);
    });
    auto region = allocator->allocate(8);
    const auto& allocation = region->get_allocation();
    auto* mapped = static_cast<const uint8_t*>(allocation->mapped_data);
    OPENVINO_ASSERT(mapped != nullptr, "The non-coherent test requires an active mapping");
    // Exercise the non-coherent policy on devices with coherent mappings or single-byte atoms.
    allocation->memory_flags &= ~static_cast<VkMemoryPropertyFlags>(VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    allocation->non_coherent_atom_size = atom_size;
    vulkan_buffer buffer(&engine, byte_layout(static_cast<int64_t>(atom_size)), region, 0, nullptr);
    buffer.fill(stream);
    check_disjoint_locks(engine, buffer, stream, mapped);
    const auto* whole_atom = buffer.lock(stream, cldnn::mem_lock_type::read);
    buffer.unlock(stream);
    OPENVINO_ASSERT(whole_atom == mapped, "An entire non-coherent atom must retain direct mapping access");
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
            return detail::allocate_vulkan_buffer(engine, size, vulkan_buffer_memory_usage::host_staging);
        });
        auto region = allocator->allocate(requested);
        OPENVINO_ASSERT(attempted.size() == (fail_pooled_allocation ? 2U : 1U) && attempted.back() == padded && region->get_size() == padded,
                        "Allocation fallback discarded transfer padding");
        unmap_for_staging(region->get_allocation());
        vulkan_buffer buffer(&engine, byte_layout(requested), region, 0, nullptr);
        constexpr uint8_t pattern = 0x5A;
        buffer.fill(stream, pattern);
        std::array<uint8_t, requested> output{};
        buffer.copy_to(stream, output.data(), 0, 0, output.size(), true);
        std::array<uint8_t, requested> expected;
        expected.fill(pattern);
        OPENVINO_ASSERT(output == expected, "The padded fallback did not preserve every staged byte");
    };
    check_fallback(requested, false);
    check_fallback(preferred, true);

    std::vector<size_t> attempted;
    vulkan_memory_allocator allocator(sizeof(uint32_t), padded, preferred, [&](size_t size) -> vulkan_buffer_allocation::ptr {
        attempted.push_back(size);
        throw std::bad_alloc{};
    });
    try {
        allocator.allocate(requested);
        OPENVINO_THROW("The allocation failure was swallowed");
    } catch (const std::bad_alloc&) {
        OPENVINO_ASSERT(attempted == std::vector<size_t>{padded}, "An already padded request must not retry");
    }
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
    OPENVINO_ASSERT(output == expected, "Staged strided transfer changed payload or padding");
    std::array<uint8_t, 2> boundaries{1, 1};
    buffer.copy_to(stream, boundaries.data(), buffer_offset - 1, 0, 1, true);
    buffer.copy_to(stream, boundaries.data(), buffer_offset + rows * columns, 1, 1, true);
    OPENVINO_ASSERT((boundaries == std::array<uint8_t, 2>{}), "Staged upload changed adjacent bytes");
}

void check_device_copies(cldnn::engine& engine, cldnn::stream& stream) {
    using cldnn::vulkan::vulkan_buffer;
    for (const size_t count : {256, 257}) {
        auto source = engine.allocate_memory(byte_layout(count + 4), cldnn::allocation_type::device_buffer, false);
        auto destination = engine.allocate_memory(byte_layout(count + 4), cldnn::allocation_type::device_buffer, false);
        unmap_for_staging(static_cast<vulkan_buffer&>(*source).get_allocation());
        unmap_for_staging(static_cast<vulkan_buffer&>(*destination).get_allocation());
        destination->fill(stream, 0xA5);

        std::vector<uint8_t> expected(count + 4, 0xA5);
        for (size_t i = 0; i < count; ++i) {
            expected[i + 2] = static_cast<uint8_t>(i);
        }
        source->copy_from(stream, expected.data() + 2, 0, 1, count, true);
        source->copy_to(stream, *destination, 1, 2, count, true);
        std::vector<uint8_t> result(expected.size());
        destination->copy_to(stream, result.data(), 0, 0, result.size(), true);
        OPENVINO_ASSERT(result == expected, "Device copy changed the payload or adjacent bytes");
    }
}

}  // namespace

int main() {
    try {
        const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
        OPENVINO_ASSERT(!devices.empty(), "No Vulkan GPU detected");
        auto engine = cldnn::engine::create(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan, devices.begin()->second);
        auto buffer = engine->allocate_memory(byte_layout(4096), cldnn::allocation_type::device_buffer, false);
        unmap_for_staging(static_cast<cldnn::vulkan::vulkan_buffer&>(*buffer).get_allocation());
        auto& stream = engine->get_service_stream();
        buffer->fill(stream);
        check_disjoint_locks(*engine, *buffer, stream);
        check_strided_rows(*buffer, stream);
        check_device_copies(*engine, stream);
        auto& vulkan_engine = static_cast<cldnn::vulkan::vulkan_engine&>(*engine);
        check_mapped_non_coherent_locks(vulkan_engine, stream);
        check_padded_allocation_fallbacks(vulkan_engine, stream);
        std::cout << "Vulkan staged memory checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
