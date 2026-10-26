//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "level_zero_allocator.hpp"
#include "allocator_core.hpp"
#include "composable_allocators.hpp"

#include <ze_api.h>

#include <cstddef>
#include <cstdint>

intel_npu::vm::LevelZeroAllocator::LevelZeroAllocator(ze_context_handle_t context): _context(context) {
}

intel_npu::vm::Block intel_npu::vm::LevelZeroAllocator::allocate(size_t bytes) {
    if (_context == nullptr || bytes == 0) {
        return Block{};
    }

    ze_host_mem_alloc_desc_t desc{ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr, 0};
    void* data = nullptr;
    // Page-aligned so device buffers satisfy the NPU's host-memory alignment expectation.
    if (zeMemAllocHost(_context, &desc, bytes, ALIGNMENT, &data) != ZE_RESULT_SUCCESS) {
        return Block{};
    }
    return Block{static_cast<uint8_t*>(data), bytes};
}

void intel_npu::vm::LevelZeroAllocator::deallocate(Block block) noexcept {
    if (_context != nullptr && block.ptr != nullptr) {
        zeMemFree(_context, block.ptr);
    }
}

intel_npu::vm::AnyAllocator intel_npu::vm::makeLevelZeroAllocator(ze_context_handle_t context) {
    // GrowingArena of BumpRegions over Level Zero memory, page-aligned so buffers are device-visible.
    return AnyAllocator{
            GrowingArena<LevelZeroAllocator, BumpRegion<LevelZeroAllocator::ALIGNMENT>,
                         DEFAULT_ARENA_INITIAL_SLAB_BYTES, DEFAULT_ARENA_MAX_SLAB_BYTES>(LevelZeroAllocator(context))};
}
