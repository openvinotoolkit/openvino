//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "allocator_core.hpp"

#include <cstddef>
#include <cstdint>
#include <new>

namespace intel_npu::vm::test {

// Test-owned counters; CountingLeaf holds a pointer to them so they survive the leaf being moved
// into a composition or erased into a BufferManager.
struct AllocCounters {
    int live = 0;        // outstanding successful allocations (successes minus frees)
    int allocCalls = 0;  // total allocate() invocations, whether they succeed or fail
};

// Host new[]/delete[] allocator leaf that records live allocations and allocate() calls.
class CountingLeaf {
    AllocCounters* _counters;

public:
    // Chunk-base alignment this backing guarantees: new[] returns max_align_t-aligned storage.
    static constexpr size_t ALIGNMENT = alignof(std::max_align_t);

    explicit CountingLeaf(AllocCounters* counters): _counters(counters) {
    }
    Block allocate(size_t bytes) {
        ++_counters->allocCalls;  // every call, success or failure
        // Mirror HostAllocator/BumpRegion: a zero-byte request is not a successful allocation. new uint8_t[0]
        // may return a unique non-null pointer, which would bump `live` and yield a truthy Block, diverging
        // from the production leaf. Reject it up front so the test leaf encodes the same zero-allocation behavior.
        if (bytes == 0) {
            return Block{};
        }
        auto* ptr = new (std::nothrow) uint8_t[bytes];
        if (ptr == nullptr) {
            return Block{};
        }
        ++_counters->live;
        return Block{ptr, bytes};
    }
    void deallocate(Block block) noexcept {
        if (block.ptr != nullptr) {
            --_counters->live;
            delete[] block.ptr;
        }
    }
    // A heap leaf tracks no bounds; it claims any non-null block, mirroring delete[] in deallocate().
    bool owns(Block block) const noexcept {
        return block.ptr != nullptr;
    }
    void recycle() noexcept {
    }
    void release() noexcept {
    }
};

}  // namespace intel_npu::vm::test
