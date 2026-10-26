//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "composable_allocators.hpp"

#include "counting_leaf.hpp"
#include "host_allocator.hpp"

#include <intel_npu/utils/utils.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace intel_npu::vm;
using intel_npu::vm::test::AllocCounters;
using intel_npu::vm::test::CountingLeaf;

namespace {

constexpr size_t kPage = intel_npu::utils::STANDARD_PAGE_SIZE;

// Owns a heap chunk for a BumpRegion under test; the region borrows the chunk and never frees it (the
// GrowingArena owns chunk lifetime in production).
class OwnedChunk {
    std::vector<uint8_t> _storage;

public:
    explicit OwnedChunk(size_t bytes): _storage(bytes) {
    }
    Block block() noexcept {
        return Block{_storage.data(), _storage.size()};
    }
};

}  // namespace

// ============================================================
// BumpRegion
// ============================================================

TEST(BumpRegionTest, HandsOutAlignedOffsets) {
    OwnedChunk chunk(8 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block first = region.allocate(100);
    const Block second = region.allocate(100);
    ASSERT_TRUE(first);
    ASSERT_TRUE(second);
    EXPECT_EQ(first.size, 100u);  // Block.size mirrors the request; alignment padding is private
    // Sub-blocks are spaced by the alignment, so each is aligned relative to the chunk base.
    EXPECT_EQ(static_cast<size_t>(second.ptr - first.ptr), kPage);
}

TEST(BumpRegionTest, RejectsRequestsThatDoNotFit) {
    OwnedChunk chunk(2 * kPage);
    BumpRegion<kPage> region(chunk.block());

    EXPECT_FALSE(region.allocate(3 * kPage));  // larger than the whole chunk (a region cannot grow it)
    EXPECT_TRUE(region.allocate(kPage));
    EXPECT_TRUE(region.allocate(kPage));
    EXPECT_FALSE(region.allocate(1));  // chunk now exhausted
}

TEST(BumpRegionTest, DeallocateRewindsMostRecentBlock) {
    OwnedChunk chunk(4 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(64);
    const Block b = region.allocate(64);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);

    region.deallocate(b);  // b is on top of the stack, so its slot is reclaimed (LIFO bump-back)
    const Block reborn = region.allocate(64);
    EXPECT_EQ(reborn.ptr, b.ptr);  // the just-freed slot is handed out again
}

TEST(BumpRegionTest, DeallocateOfAnOlderBlockIsIgnored) {
    OwnedChunk chunk(4 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(64);
    const Block b = region.allocate(64);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);

    region.deallocate(a);  // a is NOT on top (b is): a bump region cannot punch a hole, so this no-ops
    const Block c = region.allocate(64);
    EXPECT_NE(c.ptr, a.ptr);          // a's slot is not reused
    EXPECT_EQ(c.ptr, b.ptr + kPage);  // c is handed out past b, not into a's hole
}

TEST(BumpRegionTest, DeallocateIsMultiLevelLifo) {
    OwnedChunk chunk(4 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(64);
    const Block b = region.allocate(64);
    const Block c = region.allocate(64);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    ASSERT_TRUE(c);

    region.deallocate(c);  // popping c restores the cursor so b becomes the new top...
    region.deallocate(b);  // ...and can be popped too
    const Block reborn = region.allocate(64);
    EXPECT_EQ(reborn.ptr, b.ptr);  // b's slot (the new top) is reused
}

TEST(BumpRegionTest, OwnsReportsChunkMembership) {
    OwnedChunk chunk(4 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(64);
    ASSERT_TRUE(a);
    EXPECT_TRUE(region.owns(a));  // handed out of this chunk

    OwnedChunk other(kPage);
    EXPECT_FALSE(region.owns(other.block()));  // a block from a different chunk
    EXPECT_FALSE(region.owns(Block{}));        // the null block
}

TEST(BumpRegionTest, RecycleRewindsAndReusesSameMemory) {
    OwnedChunk chunk(4 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(64);
    region.allocate(64);
    ASSERT_TRUE(a);

    region.recycle();

    const Block reborn = region.allocate(64);
    EXPECT_EQ(reborn.ptr, a.ptr);  // same chunk, cursor rewound
}

TEST(BumpRegionTest, AlignmentControlsSpacing) {
    // A smaller alignment packs sub-blocks tighter.
    OwnedChunk chunk(kPage);
    BumpRegion<64> region(chunk.block());

    const Block a = region.allocate(100);
    const Block b = region.allocate(100);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_EQ(static_cast<size_t>(b.ptr - a.ptr), 128u);  // 100 rounded up to a multiple of 64
}

TEST(BumpRegionTest, RefusesWhenAlignedFootprintExceedsChunkEvenIfRawSizeFits) {
    // The raw 100 bytes fit the 100-byte chunk, but the aligned footprint (alignUp(100, 64) == 128) does
    // not, so the region refuses rather than clamping the cursor to the chunk end.
    OwnedChunk chunk(100);
    BumpRegion<64> region(chunk.block());

    EXPECT_FALSE(region.allocate(100));
    EXPECT_TRUE(region.allocate(64));  // a request whose aligned footprint fits still succeeds
}

TEST(BumpRegionTest, DeallocateReclaimsBlockThatFillsChunkToTheEnd) {
    // A block whose footprint reaches exactly the chunk end is still the top of stack, so deallocate() pops
    // it cleanly -- the last-in-chunk case a clamping cursor would miss.
    OwnedChunk chunk(2 * kPage);
    BumpRegion<kPage> region(chunk.block());

    const Block a = region.allocate(kPage);
    const Block b = region.allocate(kPage);  // fills the chunk to the end
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_FALSE(region.allocate(1));  // chunk is full

    region.deallocate(b);
    const Block reborn = region.allocate(kPage);
    EXPECT_EQ(reborn.ptr, b.ptr);  // b's slot reclaimed and handed out again
}

// ============================================================
// GrowingArena
// ============================================================

TEST(GrowingArenaTest, GrowsAcrossSlabsToFitWorkingSet) {
    // Initial slab holds one page; a second allocation forces a new slab.
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/4 * kPage>
            arena(HostAllocator{});

    const Block a = arena.allocate(kPage);
    const Block b = arena.allocate(kPage);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_NE(b.ptr, a.ptr + kPage);  // distinct slabs, not contiguous
}

TEST(GrowingArenaTest, FirstFitReusesAnEarlierSlabTail) {
    // A 3-page request leaves 1 page in slab 0; a 2-page request grows slab 1; a later 1-page request
    // must first-fit back into slab 0's tail rather than strand it.
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/4 * kPage,
                 /*maxSlabBytes=*/8 * kPage>
            arena(HostAllocator{});

    const Block big = arena.allocate(3 * kPage);
    ASSERT_TRUE(big);
    const Block grown = arena.allocate(2 * kPage);
    ASSERT_TRUE(grown);
    const Block small = arena.allocate(kPage);
    ASSERT_TRUE(small);

    EXPECT_EQ(small.ptr, big.ptr + 3 * kPage);  // packed right after `big` in slab 0
}

TEST(GrowingArenaTest, OversizeRequestGetsASlabLargeEnough) {
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/2 * kPage>
            arena(HostAllocator{});

    const Block big = arena.allocate(5 * kPage);  // larger than initial and max slab size
    ASSERT_TRUE(big);
    EXPECT_EQ(big.size, 5u * kPage);
}

TEST(GrowingArenaTest, OversizeRequestWithUnalignedSizeStillSucceeds) {
    // An oversize request whose size is not alignment-aligned must still get a chunk big enough for its
    // aligned footprint: the arena rounds the chunk size up so the strict-fit sub-allocator accepts it.
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/2 * kPage>
            arena(HostAllocator{});

    const Block big = arena.allocate(5 * kPage + 1);  // oversize and not page-aligned
    ASSERT_TRUE(big);
    EXPECT_EQ(big.size, 5u * kPage + 1);
}

TEST(GrowingArenaTest, RecycleRewindsAllSlabsAndKeepsThem) {
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/4 * kPage>
            arena(HostAllocator{});

    const Block a = arena.allocate(kPage);
    const Block b = arena.allocate(kPage);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);

    arena.recycle();  // grow & keep: rewind cursors, retain slabs

    const Block a2 = arena.allocate(kPage);
    const Block b2 = arena.allocate(kPage);
    EXPECT_EQ(a2.ptr, a.ptr);
    EXPECT_EQ(b2.ptr, b.ptr);
}

TEST(GrowingArenaTest, AdaptsToPeakThenStopsAllocatingFromParent) {
    // Once the working set is reached, replaying it allocates nothing new from the parent.
    AllocCounters counters;
    GrowingArena<CountingLeaf, BumpRegion<CountingLeaf::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/4 * kPage>
            arena(CountingLeaf{&counters});

    auto runWorkingSet = [&]() {
        ASSERT_TRUE(arena.allocate(kPage));
        ASSERT_TRUE(arena.allocate(kPage));
        ASSERT_TRUE(arena.allocate(kPage));
    };

    runWorkingSet();  // infer 1: grows slabs from the parent
    const int slabsAfterFirst = counters.allocCalls;
    EXPECT_GT(slabsAfterFirst, 0);

    for (int infer = 0; infer < 5; ++infer) {
        arena.recycle();
        runWorkingSet();
    }
    // No new parent allocations after the first working set.
    EXPECT_EQ(counters.allocCalls, slabsAfterFirst);
    EXPECT_EQ(counters.live, slabsAfterFirst);  // all slabs retained
}

TEST(GrowingArenaTest, ReleaseFreesAllSlabsThenReallocatesFromParent) {
    AllocCounters counters;
    GrowingArena<CountingLeaf, BumpRegion<CountingLeaf::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/4 * kPage>
            arena(CountingLeaf{&counters});

    ASSERT_TRUE(arena.allocate(kPage));
    ASSERT_TRUE(arena.allocate(kPage));  // forces a second slab
    const int slabsBefore = counters.live;
    EXPECT_GT(slabsBefore, 0);

    arena.release();
    EXPECT_EQ(counters.live, 0);  // every slab freed back to the parent

    // After release the arena re-mints from the parent again (growth rewound to the initial size).
    ASSERT_TRUE(arena.allocate(kPage));
    EXPECT_GT(counters.live, 0);
    EXPECT_GT(counters.allocCalls, slabsBefore);
}

TEST(GrowingArenaTest, HandsOutBlocksAlignedToBackingAlignment) {
    // Absolute alignment, not just relative spacing: the handed-out address is aligned only because the
    // backing aligns the chunk base. Odd request sizes would expose a regression in either layer.
    constexpr size_t kAlign = HostAllocator::ALIGNMENT;
    GrowingArena<HostAllocator, BumpRegion<kAlign>, /*initialSlabBytes=*/kPage, /*maxSlabBytes=*/4 * kPage> arena(
            HostAllocator{});

    for (const size_t bytes : {size_t{1}, size_t{7}, size_t{100}, size_t{kPage + 3}}) {
        const Block block = arena.allocate(bytes);
        ASSERT_TRUE(block);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(block.ptr) % kAlign, 0u);  // aligned base, not merely spaced
    }
}

TEST(GrowingArenaTest, OwnsReportsMembershipAcrossSlabs) {
    // Single-page slabs force two of them, so owns() must span every slab rather than just the latest.
    GrowingArena<HostAllocator, BumpRegion<HostAllocator::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                 /*maxSlabBytes=*/kPage>
            arena(HostAllocator{});

    const Block a = arena.allocate(kPage);  // slab 0
    const Block b = arena.allocate(kPage);  // forces slab 1
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_TRUE(arena.owns(a));  // carved from slab 0
    EXPECT_TRUE(arena.owns(b));  // carved from slab 1

    OwnedChunk foreign(kPage);
    EXPECT_FALSE(arena.owns(foreign.block()));  // memory the arena never handed out
    EXPECT_FALSE(arena.owns(Block{}));          // the null block

    arena.release();
    EXPECT_FALSE(arena.owns(a));  // no slabs remain to own anything
}

// ============================================================
// AnyAllocator (type erasure)
// ============================================================

TEST(AnyAllocatorTest, OwnsDelegatesToTheErasedAllocator) {
    AllocCounters counters;
    AnyAllocator allocator{CountingLeaf{&counters}};

    const Block block = allocator.allocate(kPage);
    ASSERT_TRUE(block);
    EXPECT_TRUE(allocator.owns(block));     // a leaf claims any non-null block it could free
    EXPECT_FALSE(allocator.owns(Block{}));  // ...but never the null block

    allocator.deallocate(block);
}

TEST(AnyAllocatorTest, EmptyHandleOwnsNothing) {
    uint8_t byte = 0;
    AnyAllocator allocator;  // default-constructed: no erased allocator behind it
    ASSERT_FALSE(allocator);
    EXPECT_FALSE(allocator.owns(Block{&byte, 1}));  // an empty handle owns nothing, even a non-null block
}
