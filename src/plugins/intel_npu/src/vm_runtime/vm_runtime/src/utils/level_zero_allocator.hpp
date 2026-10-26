//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "allocator_core.hpp"
#include "vm_export.hpp"

#include <intel_npu/utils/utils.hpp>

#include <ze_api.h>

#include <cstddef>
#include <cstdint>

namespace intel_npu::vm {

// Tuning for the default GrowingArena: first slab size and the geometric per-slab growth cap. It grows on
// demand (doubling each slab toward the cap) and reuses slabs, so the initial size need not match the working
// set. Kept small so a first allocation of a few small buffers (e.g. output-shape prediction / metadata) does
// not eagerly reserve tens of MiB of device memory; a larger first request still gets a chunk sized to it.
inline constexpr uint64_t DEFAULT_ARENA_INITIAL_SLAB_BYTES = uint64_t{1} << 20;  // 1 MiB
inline constexpr uint64_t DEFAULT_ARENA_MAX_SLAB_BYTES = uint64_t{64} << 20;     // 64 MiB

// Leaf over Level Zero host-visible memory (zeMemAllocHost/zeMemFree) so the NPU can read the buffers
// directly. Bound to one context. recycle()/release() are no-ops: it retains nothing and frees per-block.
class NPU_VM_EXPORT LevelZeroAllocator {
    ze_context_handle_t _context;

public:
    // Chunk-base alignment this backing guarantees: zeMemAllocHost is asked for page-aligned memory.
    static constexpr size_t ALIGNMENT = intel_npu::utils::STANDARD_PAGE_SIZE;

    explicit LevelZeroAllocator(ze_context_handle_t context);

    // Returns a null Block on failure (never throws).
    Block allocate(size_t bytes);
    void deallocate(Block block) noexcept;
};

// Level Zero device allocator: a GrowingArena of BumpRegions over a Level Zero backing, page-aligned so
// buffers are device-visible. Slabs are reserved lazily, so the context is touched only on first allocation.
NPU_VM_EXPORT AnyAllocator makeLevelZeroAllocator(ze_context_handle_t context);

}  // namespace intel_npu::vm
