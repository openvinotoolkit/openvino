//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "host_allocator.hpp"
#include "allocator_core.hpp"
#include "composable_allocators.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

intel_npu::vm::Block intel_npu::vm::HostAllocator::allocate(size_t bytes) {
    // operator new[] takes size_t, so buffers whose size does not fit in size_t cannot be allocated on this host.
    static_assert(sizeof(size_t) >= sizeof(int64_t), "Bytecode buffer allocation requires a 64-bit size_t");
    // Reject empty requests up front: new[] may hand back a unique non-null pointer for a zero-size
    // allocation, which would make the returned Block evaluate true. Mirror BumpRegion/LevelZeroAllocator
    // and report a null Block so `if (block)` stays a reliable success check.
    if (bytes == 0) {
        return Block{};
    }
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    auto* ptr = new (std::nothrow) uint8_t[bytes];
    if (ptr == nullptr) {
        return Block{};
    }
    return Block{ptr, bytes};
}

void intel_npu::vm::HostAllocator::deallocate(Block block) noexcept {
    if (block.ptr == nullptr) {
        return;
    }
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
    delete[] block.ptr;
}

intel_npu::vm::AnyAllocator intel_npu::vm::makeHostAllocator() {
    // GrowingArena of BumpRegions over the host heap. No device-visibility requirement, so sub-blocks need
    // only natural alignment.
    return AnyAllocator{
            GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, DEFAULT_HOST_ARENA_INITIAL_SLAB_BYTES,
                         DEFAULT_HOST_ARENA_MAX_SLAB_BYTES>(HostAllocator{})};
}
