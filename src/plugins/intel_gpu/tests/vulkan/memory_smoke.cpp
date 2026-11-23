// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstdint>
#include <iostream>

#include "intel_gpu/runtime/device_query.hpp"
#include "openvino/core/except.hpp"
#include "vulkan/vulkan_engine.hpp"
#include "vulkan/vulkan_memory.hpp"

namespace {

cldnn::layout byte_layout(int64_t count) {
    return {ov::PartialShape{1, 1, 1, count}, ov::element::u8, cldnn::format::bfyx};
}

void check_disjoint_locks(cldnn::engine& engine, cldnn::memory& buffer, cldnn::stream& stream) {
    auto first = engine.create_subbuffer(buffer, byte_layout(2), 1);
    auto second = engine.create_subbuffer(buffer, byte_layout(2), 3);
    auto* first_data = static_cast<uint8_t*>(first->lock(stream, cldnn::mem_lock_type::write));
    auto* second_data = static_cast<uint8_t*>(second->lock(stream, cldnn::mem_lock_type::read_write));
    first_data[0] = 11;
    first_data[1] = 12;
    second_data[0] = 21;
    second_data[1] = 22;
    first->unlock(stream);
    second->unlock(stream);

    std::array<uint8_t, 8> actual{};
    buffer.copy_to(stream, actual.data(), 0, 0, actual.size(), true);
    constexpr std::array<uint8_t, 8> expected{0, 11, 12, 21, 22, 0, 0, 0};
    OPENVINO_ASSERT(actual == expected, "Unlocking a staged subbuffer lost a write to a disjoint view");
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

}  // namespace

int main() {
    try {
        const auto devices = cldnn::device_query(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan).get_available_devices();
        OPENVINO_ASSERT(!devices.empty(), "No Vulkan GPU detected");
        auto engine = cldnn::engine::create(cldnn::engine_types::vulkan, cldnn::runtime_types::vulkan, devices.begin()->second);
        auto buffer = engine->allocate_memory(byte_layout(4096), cldnn::allocation_type::device_buffer, false);
        const auto& allocation = static_cast<cldnn::vulkan::vulkan_buffer&>(*buffer).get_allocation();
        if (allocation->is_mapped()) {
            // Unmap this test buffer to exercise real staging copies on unified-memory GPUs.
            vkUnmapMemory(allocation->device, allocation->memory);
            allocation->mapped_data = nullptr;
        }
        auto& stream = engine->get_service_stream();
        buffer->fill(stream);
        check_disjoint_locks(*engine, *buffer, stream);
        check_strided_rows(*buffer, stream);
        std::cout << "Vulkan staged memory checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
