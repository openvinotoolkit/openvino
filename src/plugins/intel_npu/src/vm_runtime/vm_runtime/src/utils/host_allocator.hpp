//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "allocator_core.hpp"
#include "vm_export.hpp"

#include <cstddef>
#include <cstdint>

namespace intel_npu::vm {

// Leaf over the host heap (new[]/delete[]), used as a GrowingArena backing by makeHostAllocator.
// recycle()/release() are no-ops: the heap retains nothing and frees per-block in deallocate().
class NPU_VM_EXPORT HostAllocator {
public:
    // Chunk-base alignment this backing guarantees: new[] returns max_align_t-aligned storage.
    static constexpr size_t ALIGNMENT = alignof(std::max_align_t);

    Block allocate(size_t bytes);
    void deallocate(Block block) noexcept;
};

// Tuning for the default host GrowingArena. It grows on demand (doubling each slab toward the cap) and reuses
// slabs, so the initial size need not match the working set. Kept small so a first allocation of a few small
// buffers (e.g. output-shape prediction / metadata) does not eagerly reserve tens of MiB; a larger first
// request still gets a chunk sized to it.
inline constexpr uint64_t DEFAULT_HOST_ARENA_INITIAL_SLAB_BYTES = uint64_t{1} << 20;  // 1 MiB
inline constexpr uint64_t DEFAULT_HOST_ARENA_MAX_SLAB_BYTES = uint64_t{64} << 20;     // 64 MiB

// Host allocator: a GrowingArena of BumpRegions over the host heap, mirroring makeLevelZeroAllocator's
// shape. Used on the no-device path (output-shape prediction / offline execute) and by the unit tests.
NPU_VM_EXPORT AnyAllocator makeHostAllocator();

}  // namespace intel_npu::vm
