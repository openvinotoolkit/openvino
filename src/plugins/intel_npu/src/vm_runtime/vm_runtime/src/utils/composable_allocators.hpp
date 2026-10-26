//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "allocator_core.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

namespace intel_npu::vm {

// Rounds up to the next multiple of `alignment`: sizes a block's footprint so the next block starts aligned.
// Returns 0 to signal overflow (no multiple of `alignment` is representable >= `value`), so callers refuse the
// allocation instead of proceeding with a footprint that wrapped to a smaller multiple (which would walk the
// bump cursor backwards). Callers guard `bytes == 0` before calling, so a 0 result is unambiguously the sentinel.
inline size_t alignUp(size_t value, size_t alignment) noexcept {
    if (alignment <= 1) {
        return value;
    }
    if (value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
        return 0;  // overflow: no representable aligned footprint; callers treat 0 as failure
    }
    return (value + alignment - 1) / alignment * alignment;
}

// Carves aligned sub-blocks out of one fixed chunk it does NOT own. `Alignment` is the sub-block alignment,
// advertised as a `static constexpr` so a composing parent (GrowingArena) can size chunks to it.
template <size_t Alignment>
class BumpRegion {
    static_assert(Alignment >= 1, "BumpRegion: Alignment must be at least 1");

    Block _chunk;    // borrowed; the arena owns this chunk and frees it
    size_t _high{};  // next free offset
public:
    static constexpr size_t ALIGNMENT = Alignment;

    explicit BumpRegion(Block chunk): _chunk(chunk) {
    }

    Block allocate(size_t bytes) {
        if (bytes == 0 || !_chunk) {
            return Block{};
        }
        const size_t start = _high;
        // Reserve the aligned footprint so the next block starts aligned. alignUp returns 0 if that footprint
        // overflows size_t; refusing it (rather than clamping to the chunk end) keeps `_high` on a footprint
        // boundary, so deallocate() can pop even the last block in the chunk. A non-zero footprint is never
        // < bytes, so the same bound check covers the raw-size check too.
        const size_t footprint = alignUp(bytes, Alignment);
        if (footprint == 0 || footprint > _chunk.size - start) {
            return Block{};
        }
        _high = start + footprint;
        return Block{std::next(_chunk.ptr, static_cast<std::ptrdiff_t>(start)), bytes};
    }

    // Pops a block only if it is the most recent hand-out: its footprint reaches the cursor.
    void deallocate(Block block) noexcept {
        if (!owns(block)) {
            return;
        }
        const auto start = static_cast<size_t>(block.ptr - _chunk.ptr);
        if (start + alignUp(block.size, Alignment) == _high) {
            _high = start;
        }
    }

    bool owns(Block block) const noexcept {
        return block.ptr != nullptr && _chunk.ptr != nullptr && block.ptr >= _chunk.ptr &&
               block.ptr < std::next(_chunk.ptr, static_cast<std::ptrdiff_t>(_chunk.size));
    }

    void recycle() noexcept {
        _high = 0;  // rewind; the chunk is kept for reuse next inference
    }
};

// Grows a set of sub-allocators, each carving one chunk from a single `Backing`. `InitialSlabBytes`/
// `MaxSlabBytes` tune slab growth; alignment comes from `SubAllocator::ALIGNMENT`, not a parameter.
template <class Backing, class SubAllocator, size_t InitialSlabBytes, size_t MaxSlabBytes>
class GrowingArena {
    static_assert(InitialSlabBytes > 0, "GrowingArena: InitialSlabBytes must be greater than 0");

    // The only alignment the arena needs: to size each chunk to the sub-allocator's footprint. The chunk
    // base alignment is the backing's responsibility.
    static constexpr size_t ALIGNMENT = SubAllocator::ALIGNMENT;

    // Sub-block offsets are multiples of SubAllocator::ALIGNMENT, but absolute alignment holds only if every
    // chunk base is too -- so Backing::ALIGNMENT must be a multiple of it, else carved blocks are silently
    // under-aligned.
    static_assert(Backing::ALIGNMENT % SubAllocator::ALIGNMENT == 0,
                  "GrowingArena: Backing::ALIGNMENT must be a multiple of SubAllocator::ALIGNMENT, "
                  "otherwise carved sub-blocks are not aligned to SubAllocator::ALIGNMENT");

public:
    explicit GrowingArena(Backing backing): _backing(std::move(backing)), _nextSlabBytes(InitialSlabBytes) {
    }

    // Move-only and non-reassignable. AnyAllocator stores the arena by value, so it must be move-constructible;
    // the move constructor is written out (rather than `= default`) so all five special members are explicitly
    // user-provided. Assignment is deleted because a memberwise move would overwrite `_slabs` without first
    // freeing the destination's chunks, leaking them; copying an owning arena is meaningless.
    GrowingArena(GrowingArena&& other) noexcept
            : _backing(std::move(other._backing)),
              _nextSlabBytes(other._nextSlabBytes),
              _slabs(std::move(other._slabs)) {
    }
    GrowingArena(const GrowingArena&) = delete;
    GrowingArena& operator=(GrowingArena&&) = delete;
    GrowingArena& operator=(const GrowingArena&) = delete;

    ~GrowingArena() {
        freeChunks();  // the arena owns the chunks; nothing else frees them
    }

    Block allocate(size_t bytes) {
        if (bytes == 0) {
            return Block{};
        }
        // First-fit so a small request can reuse the tail of an earlier slab a larger one skipped.
        for (auto& slab : _slabs) {
            if (Block block = slab.subAllocator.allocate(bytes)) {
                return block;
            }
        }
        // Size the chunk to the aligned footprint, not raw `bytes`: the sub-allocator reserves that footprint
        // and would refuse a chunk merely `bytes` long when `bytes` is not a multiple of the alignment.
        const size_t footprint = alignUp(bytes, ALIGNMENT);
        if (footprint == 0) {
            return Block{};  // footprint overflowed size_t; no chunk can satisfy the request
        }
        const size_t want = std::max(footprint, _nextSlabBytes);
        Block chunk = _backing.allocate(want);
        if (!chunk) {
            return Block{};
        }
        try {
            SubAllocator subAllocator{chunk};
            Block block = subAllocator.allocate(bytes);
            if (!block) {
                _backing.deallocate(chunk);  // a fresh chunk could not satisfy the request
                return Block{};
            }
            _slabs.push_back(Slab{chunk, std::move(subAllocator)});
            growNextSlabBytes();
            return block;
        } catch (...) {
            _backing.deallocate(chunk);  // don't leak the chunk if the bookkeeping push throws
            return Block{};
        }
    }

    void deallocate(Block block) noexcept {
        for (auto& slab : _slabs) {
            if (slab.subAllocator.owns(block)) {
                slab.subAllocator.deallocate(block);
                return;
            }
        }
    }

    // True if the block was carved from one of this arena's chunks (i.e. some sub-allocator owns it).
    bool owns(Block block) const noexcept {
        for (const auto& slab : _slabs) {
            if (slab.subAllocator.owns(block)) {
                return true;
            }
        }
        return false;
    }

    void recycle() noexcept {
        for (auto& slab : _slabs) {
            slab.subAllocator.recycle();  // rewind, keep the chunk for reuse
        }
    }

    void release() noexcept {
        freeChunks();
        _slabs.clear();
        _nextSlabBytes = InitialSlabBytes;
    }

private:
    struct Slab {
        Block chunk;                // allocated from _backing
        SubAllocator subAllocator;  // carves `chunk`; owns no memory of its own
    };

    void freeChunks() noexcept {
        for (auto& slab : _slabs) {
            _backing.deallocate(slab.chunk);
        }
    }

    void growNextSlabBytes() noexcept {
        if (MaxSlabBytes != 0 && _nextSlabBytes >= MaxSlabBytes) {
            _nextSlabBytes = MaxSlabBytes;
            return;
        }
        if (_nextSlabBytes > std::numeric_limits<size_t>::max() / 2) {
            return;  // overflow guard: keep the current size
        }
        const size_t doubled = _nextSlabBytes * 2;
        _nextSlabBytes = (MaxSlabBytes != 0) ? std::min(doubled, MaxSlabBytes) : doubled;
    }

    Backing _backing;          // sole owner of chunk memory; chunks are allocated and freed through it
    size_t _nextSlabBytes;     // next slab size; starts at InitialSlabBytes, grows toward MaxSlabBytes
    std::vector<Slab> _slabs;  // chunk + the sub-allocator carving it, in growth order
};

}  // namespace intel_npu::vm
