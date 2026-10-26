//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffer.hpp"
#include "buffer_metadata.hpp"
#include "composable_allocators.hpp"
#include "counting_leaf.hpp"
#include "host_allocator.hpp"

#include <intel_npu/utils/utils.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace intel_npu::vm;
using intel_npu::vm::test::AllocCounters;
using intel_npu::vm::test::CountingLeaf;

// ============================================================
// Buffer Tests
// ============================================================

class VirtualMachineBufferTest : public ::testing::Test {};

TEST_F(VirtualMachineBufferTest, ConstructNewAllocation) {
    BufferManager mgr(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeHostAllocator());
    {
        // 64 bytes + Read-Write
        const auto h = mgr.create(/*size=*/64, Permission::ReadWrite);
        auto& buf = mgr.getBuffer(h);
        EXPECT_NE(buf.getData(), nullptr);
        EXPECT_EQ(buf.getSize(), 64u);
        EXPECT_EQ(buf.getPermission(), Permission::ReadWrite);
        EXPECT_EQ(buf.getOwnership(), Ownership::Owned);
    }
    {
        // 128 bytes + Read-Only
        const auto h = mgr.create(/*size=*/128, Permission::Read);
        auto& buf = mgr.getBuffer(h);
        EXPECT_NE(buf.getData(), nullptr);
        EXPECT_EQ(buf.getSize(), 128u);
        EXPECT_EQ(buf.getPermission(), Permission::Read);
        EXPECT_EQ(buf.getOwnership(), Ownership::Owned);
    }
}

TEST_F(VirtualMachineBufferTest, ConstructFromExternalMemory) {
    std::array<uint8_t, 8> externalData = {1, 2, 3, 4, 5, 6, 7, 8};
    Buffer buf(externalData.data(), externalData.size(), Permission::Read);
    EXPECT_EQ(buf.getData(), externalData.data());
    EXPECT_EQ(buf.getSize(), externalData.size());
    EXPECT_EQ(buf.getPermission(), Permission::Read);
    EXPECT_EQ(buf.getOwnership(), Ownership::Unowned);
}

TEST_F(VirtualMachineBufferTest, DataWrite) {
    std::array<uint8_t, 4> data = {0xDE, 0xAD, 0xBE, 0xEF};
    BufferManager mgr(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeHostAllocator());
    {
        // Read-Write buffer
        const auto h = mgr.create(4, Permission::ReadWrite);
        auto& buf = mgr.getBuffer(h);
        buf.writeData(/*offset=*/0, data.data(), data.size());
        EXPECT_EQ(buf.getData()[0], 0xDE);
        EXPECT_EQ(buf.getData()[1], 0xAD);
        EXPECT_EQ(buf.getData()[2], 0xBE);
        EXPECT_EQ(buf.getData()[3], 0xEF);
    }
    {
        // Read-only buffer
        const auto h = mgr.create(4, Permission::Read);
        auto& buf = mgr.getBuffer(h);
        EXPECT_THROW(buf.writeData(/*offset=*/0, data.data(), data.size()), std::runtime_error);
    }
    {
        // Out-of-bounds write
        const auto h = mgr.create(4, Permission::ReadWrite);
        auto& buf = mgr.getBuffer(h);
        EXPECT_THROW(buf.writeData(/*offset=*/2, data.data(), data.size()), std::out_of_range);
    }
}

TEST_F(VirtualMachineBufferTest, MoveLeavesSourceAsEmptyView) {
    std::array<uint8_t, 4> payload = {1, 2, 3, 4};
    Buffer source(payload.data(), payload.size(), Permission::ReadWrite, Ownership::Unowned);

    Buffer moved(std::move(source));
    EXPECT_EQ(moved.getData(), payload.data());
    EXPECT_EQ(moved.getSize(), payload.size());
    EXPECT_EQ(moved.getPermission(), Permission::ReadWrite);
    EXPECT_EQ(moved.getOwnership(), Ownership::Unowned);

    EXPECT_EQ(source.getData(), nullptr);
    EXPECT_EQ(source.getSize(), 0u);

    std::array<uint8_t, 2> otherPayload = {9, 9};
    Buffer target(otherPayload.data(), otherPayload.size(), Permission::Read, Ownership::Unowned);
    target = std::move(moved);

    EXPECT_EQ(target.getData(), payload.data());
    EXPECT_EQ(target.getSize(), payload.size());
    EXPECT_EQ(target.getPermission(), Permission::ReadWrite);
    EXPECT_EQ(target.getOwnership(), Ownership::Unowned);

    EXPECT_EQ(moved.getData(), nullptr);
    EXPECT_EQ(moved.getSize(), 0u);
}

// ============================================================
// BufferManager Tests
// ============================================================

class VirtualMachineBufferManagerTest : public ::testing::Test {
protected:
    static constexpr size_t DEFAULT_MAX_MEMORY = 1024;
    BufferManager manager{DEFAULT_MAX_MEMORY, makeHostAllocator()};
};

TEST_F(VirtualMachineBufferManagerTest, Create) {
    auto handle = manager.create(/*size=*/64);
    ASSERT_TRUE(manager.exists(handle));
    ASSERT_NO_THROW(manager.getBuffer(handle));
    auto& buf = manager.getBuffer(handle);
    EXPECT_NE(buf.getData(), nullptr);
    EXPECT_EQ(buf.getSize(), 64u);
    EXPECT_EQ(buf.getPermission(), Permission::ReadWrite);
    EXPECT_EQ(buf.getOwnership(), Ownership::Owned);
}

TEST_F(VirtualMachineBufferManagerTest, CreateMultipleBuffers) {
    auto handle1 = manager.create(/*size=*/64);
    auto handle2 = manager.create(/*size=*/128);
    ASSERT_TRUE(manager.exists(handle1));
    ASSERT_TRUE(manager.exists(handle2));
    EXPECT_NE(handle1, handle2);
}

TEST_F(VirtualMachineBufferManagerTest, CreateZeroSizeThrows) {
    EXPECT_THROW(manager.create(0), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, CreateExactlyMaximumSizeSucceeds) {
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(VirtualMachineBufferManagerTest, CreateExceedingMaximumSizeThrows) {
    EXPECT_THROW(manager.create(DEFAULT_MAX_MEMORY + 1), std::bad_alloc);
}

TEST_F(VirtualMachineBufferManagerTest, CreateCumulativeExceedingMaximumSizeThrows) {
    manager.create(DEFAULT_MAX_MEMORY / 2);
    EXPECT_THROW(manager.create(DEFAULT_MAX_MEMORY / 2 + 1), std::bad_alloc);
}

TEST_F(VirtualMachineBufferManagerTest, CreateAfterDeleteFitsWithinMaximumSize) {
    auto handle = manager.create(DEFAULT_MAX_MEMORY);
    manager.deleteBuffer(handle);
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromMemory) {
    std::array<uint8_t, 4> data = {0xDE, 0xAD, 0xBE, 0xEF};
    const auto handle = manager.createFromMemory(data.data(), data.size(), Permission::Read);
    EXPECT_TRUE(manager.exists(handle));
    ASSERT_NO_THROW(manager.getBuffer(handle));
    const auto& buf = manager.getBuffer(handle);
    EXPECT_EQ(buf.getData(), data.data());
    EXPECT_EQ(buf.getSize(), data.size());
    EXPECT_EQ(buf.getPermission(), Permission::Read);
    EXPECT_EQ(buf.getOwnership(), Ownership::Unowned);
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromMemoryNullDataThrows) {
    EXPECT_THROW(manager.createFromMemory(nullptr, /*size=*/4, Permission::Read), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromMemoryZeroSizeThrows) {
    std::array<uint8_t, 4> data = {};
    EXPECT_THROW(manager.createFromMemory(data.data(), 0, Permission::Read), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromMemoryDoesNotContributeToMemoryBudget) {
    // Fill budget with external memory, then still create an owned buffer
    std::array<uint8_t, DEFAULT_MAX_MEMORY * 2> externalData = {};
    manager.createFromMemory(externalData.data(), externalData.size(), Permission::Read);
    EXPECT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromBuffer) {
    auto handle = manager.create(/*size=*/64, Permission::ReadWrite);
    ASSERT_TRUE(manager.exists(handle));
    const auto& buf = manager.getBuffer(handle);
    EXPECT_NE(buf.getData(), nullptr);
    EXPECT_EQ(buf.getSize(), 64u);
    EXPECT_EQ(buf.getPermission(), Permission::ReadWrite);
    EXPECT_EQ(buf.getOwnership(), Ownership::Owned);

    auto newHandle = manager.createFromBuffer(handle, /*offset=*/32, /*size=*/32);
    ASSERT_TRUE(manager.exists(newHandle));
    const auto& newBuf = manager.getBuffer(newHandle);
    EXPECT_EQ(newBuf.getData(), buf.getData() + 32);
    EXPECT_EQ(newBuf.getSize(), 32u);
    EXPECT_EQ(newBuf.getPermission(), Permission::ReadWrite);
    EXPECT_EQ(newBuf.getOwnership(), Ownership::Unowned);
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromBufferEncodesTwoDimensionalSubviewOffsetsInBasePointer) {
    // Mirrors a 2D buffer.subview over a row-major [5, 5] i32 source with source strides [5, 1].
    // offsets [2, 2], sizes [2, 2], and subview strides [1, 1] select:
    // [12, 13]
    // [17, 18]
    std::array<int32_t, 25> sourceData = {};
    std::iota(sourceData.begin(), sourceData.end(), 0);

    const auto sourceSizeBytes = sourceData.size() * sizeof(sourceData.front());
    const auto sourceHandle =
            manager.createFromMemory(reinterpret_cast<uint8_t*>(sourceData.data()), sourceSizeBytes, Permission::Read);

    constexpr size_t sourceRowStride = 5;
    constexpr size_t sourceColStride = 1;
    constexpr size_t rowOffset = 2;
    constexpr size_t colOffset = 2;
    constexpr size_t subviewRowStride = 1;
    constexpr size_t subviewColStride = 1;

    const auto startElement = rowOffset * sourceRowStride + colOffset * sourceColStride;
    const auto byteOffset = startElement * sizeof(sourceData.front());

    const auto derivedRowStride = subviewRowStride * sourceRowStride;
    const auto derivedColStride = subviewColStride * sourceColStride;
    const auto maxDerivedElementOffset = derivedRowStride + derivedColStride;
    const auto derivedSizeBytes = (maxDerivedElementOffset + 1) * sizeof(sourceData.front());

    const auto derivedHandle = manager.createFromBuffer(sourceHandle, byteOffset, derivedSizeBytes);
    const auto& derivedBuffer = manager.getBuffer(derivedHandle);
    const auto* derivedData = reinterpret_cast<const int32_t*>(derivedBuffer.getData());

    EXPECT_EQ(derivedData[0 * derivedRowStride + 0 * derivedColStride], 12);
    EXPECT_EQ(derivedData[0 * derivedRowStride + 1 * derivedColStride], 13);
    EXPECT_EQ(derivedData[1 * derivedRowStride + 0 * derivedColStride], 17);
    EXPECT_EQ(derivedData[1 * derivedRowStride + 1 * derivedColStride], 18);
}

TEST_F(VirtualMachineBufferManagerTest, ComputeSubviewStartElementRejectsMismatchedSubviewRanks) {
    BufferMetadata sourceMeta;
    sourceMeta.shape = {5, 5};
    sourceMeta.strides = {5, 1};

    EXPECT_FALSE(computeSubviewStartElement(sourceMeta, /*offsets=*/{0, 0}, /*sizes=*/{1},
                                            /*strides=*/{1, 1})
                         .has_value());
    EXPECT_FALSE(computeSubviewStartElement(sourceMeta, /*offsets=*/{0, 0}, /*sizes=*/{1, 1},
                                            /*strides=*/{1})
                         .has_value());
    EXPECT_FALSE(computeSubviewStartElement(sourceMeta, /*offsets=*/{0}, /*sizes=*/{1, 1},
                                            /*strides=*/{1, 1})
                         .has_value());
}

TEST_F(VirtualMachineBufferManagerTest, ComputeSubviewStartElementRejectsMismatchedSourceMetadataRank) {
    BufferMetadata sourceMeta;
    sourceMeta.shape = {5};
    sourceMeta.strides = {5, 1};

    EXPECT_FALSE(computeSubviewStartElement(sourceMeta, /*offsets=*/{0, 0}, /*sizes=*/{1, 1},
                                            /*strides=*/{1, 1})
                         .has_value());

    sourceMeta.shape = {5, 5};
    sourceMeta.strides = {5};
    EXPECT_FALSE(computeSubviewStartElement(sourceMeta, /*offsets=*/{0, 0}, /*sizes=*/{1, 1},
                                            /*strides=*/{1, 1})
                         .has_value());
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromBufferInvalidHandleThrows) {
    EXPECT_THROW(manager.createFromBuffer(9999, /*offset=*/0, /*size=*/1), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, CreateFromBufferOutOfBoundsThrows) {
    auto handle = manager.create(/*size=*/64);
    ASSERT_TRUE(manager.exists(handle));

    EXPECT_THROW(manager.createFromBuffer(handle, /*offset=*/32, /*size=*/33), std::out_of_range);
    EXPECT_THROW(manager.createFromBuffer(handle, /*offset=*/64, /*size=*/1), std::out_of_range);
}

TEST_F(VirtualMachineBufferManagerTest, MultipleExternalBuffersHaveUniqueHandles) {
    std::array<uint8_t, 4> data = {};
    const auto h1 = manager.createFromMemory(data.data(), data.size(), Permission::Read);
    const auto h2 = manager.createFromMemory(data.data() + 2, data.size(), Permission::Read);
    EXPECT_NE(h1, h2);
}

TEST_F(VirtualMachineBufferManagerTest, DeleteBuffer) {
    std::array<uint8_t, 4> data = {};
    const auto handle = manager.createFromMemory(data.data(), data.size(), Permission::Read);
    manager.deleteBuffer(handle);
    EXPECT_FALSE(manager.exists(handle));
}

TEST_F(VirtualMachineBufferManagerTest, DeleteBufferInvalidHandleThrows) {
    EXPECT_THROW(manager.deleteBuffer(9999), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, DeleteSameHandleTwiceThrows) {
    const auto handle = manager.create(/*size=*/4);
    manager.deleteBuffer(handle);
    EXPECT_THROW(manager.deleteBuffer(handle), std::invalid_argument);
}

TEST_F(VirtualMachineBufferManagerTest, DeleteBufferDeletesDerivedBuffer) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);
    ASSERT_TRUE(manager.exists(derivedHandle));

    manager.deleteBuffer(parentHandle);

    EXPECT_FALSE(manager.exists(parentHandle));
    EXPECT_FALSE(manager.exists(derivedHandle));
}

TEST_F(VirtualMachineBufferManagerTest, DeleteDerivedBufferDoesNotDeleteParent) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(derivedHandle);

    EXPECT_FALSE(manager.exists(derivedHandle));
    EXPECT_TRUE(manager.exists(parentHandle));
    EXPECT_NO_THROW(manager.deleteBuffer(parentHandle));
}

TEST_F(VirtualMachineBufferManagerTest, DeleteBufferCascadesAcrossChain) {
    const auto handleA = manager.create(/*size=*/64);
    const auto handleB = manager.createFromBuffer(handleA, /*offset=*/0, /*size=*/64);
    const auto handleC = manager.createFromBuffer(handleB, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(handleA);

    EXPECT_FALSE(manager.exists(handleA));
    EXPECT_FALSE(manager.exists(handleB));
    EXPECT_FALSE(manager.exists(handleC));
}

TEST_F(VirtualMachineBufferManagerTest, DeleteDerivedBufferThenDeleteParentDoesNotThrow) {
    const auto parentHandle = manager.create(/*size=*/64);
    const auto derivedHandle = manager.createFromBuffer(parentHandle, /*offset=*/0, /*size=*/32);

    manager.deleteBuffer(derivedHandle);
    EXPECT_NO_THROW(manager.deleteBuffer(parentHandle));

    EXPECT_FALSE(manager.exists(parentHandle));
    EXPECT_FALSE(manager.exists(derivedHandle));
}

TEST_F(VirtualMachineBufferManagerTest, RecycleReleasesOwnedBuffersAndClearsBudget) {
    const auto owned = manager.create(/*size=*/256);
    std::array<uint8_t, 4> external = {};
    const auto unowned = manager.createFromMemory(external.data(), external.size(), Permission::Read);
    ASSERT_GT(manager.getCurrentMemorySize(), 0u);

    manager.recycle();

    EXPECT_EQ(manager.getCurrentMemorySize(), 0u);
    EXPECT_FALSE(manager.exists(owned));
    EXPECT_FALSE(manager.exists(unowned));
    // Handles stay monotonic across recycles: a fresh create never reuses a released handle.
    const auto reborn = manager.create(/*size=*/256);
    EXPECT_NE(reborn, owned);
}

TEST_F(VirtualMachineBufferManagerTest, RecycleLetsBudgetBeReusedAcrossManyRounds) {
    // recycle() rewinds the budget each round, so the same full-budget allocation keeps succeeding.
    for (int round = 0; round < 100; ++round) {
        ASSERT_NO_THROW(manager.create(DEFAULT_MAX_MEMORY));
        manager.recycle();
        ASSERT_EQ(manager.getCurrentMemorySize(), 0u);
    }
}

// ============================================================
// BufferManager over an arena allocator
//
// The device path wraps a GrowingArena. These tests stand in a CountingLeaf-backed arena and assert the
// BufferManager<->allocator contract: recycle() reuses memory across inferences and
// owned buffers are handed out zeroed.
// ============================================================

namespace {

constexpr size_t kPage = intel_npu::utils::STANDARD_PAGE_SIZE;

// A GrowingArena over a CountingLeaf, erased into AnyAllocator, so a test can observe backing allocations.
AnyAllocator makeCountingArena(AllocCounters& counters) {
    return AnyAllocator{GrowingArena<CountingLeaf, BumpRegion<CountingLeaf::ALIGNMENT>, /*initialSlabBytes=*/kPage,
                                     /*maxSlabBytes=*/0>(CountingLeaf{&counters})};
}

}  // namespace

TEST(VirtualMachineBufferManagerArenaTest, RecycleReusesDeviceMemoryAcrossRounds) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeCountingArena(counters));

    auto runRound = [&]() {
        manager.create(kPage);
        manager.create(kPage);
        manager.create(kPage);
    };

    runRound();  // first inference grows the arena from the parent
    const int slabsAfterFirst = counters.allocCalls;
    EXPECT_GT(slabsAfterFirst, 0);

    for (int round = 0; round < 5; ++round) {
        manager.recycle();
        EXPECT_EQ(manager.getCurrentMemorySize(), 0u);  // budget freed for reuse
        runRound();
    }
    // No new parent allocations after the first round, and slabs are retained (not freed).
    EXPECT_EQ(counters.allocCalls, slabsAfterFirst);
    EXPECT_EQ(counters.live, slabsAfterFirst);
}

TEST(VirtualMachineBufferManagerArenaTest, RecycleRetainsUnderlyingSlabsForReuse) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeCountingArena(counters));

    manager.create(kPage);
    manager.create(kPage);
    const int slabsBefore = counters.live;
    ASSERT_GT(slabsBefore, 0);

    manager.recycle();
    EXPECT_EQ(counters.live, slabsBefore);          // slabs are kept for reuse, not freed
    EXPECT_EQ(manager.getCurrentMemorySize(), 0u);  // budget cleared

    // The manager stays usable: a fresh create reuses existing slabs (no new parent allocation).
    ASSERT_NO_THROW(manager.create(kPage));
    EXPECT_EQ(counters.live, slabsBefore);  // no new slab needed
}

TEST(VirtualMachineBufferManagerArenaTest, HandsOutZeroedBuffersAfterRecycle) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeCountingArena(counters));

    const std::array<uint8_t, 8> pattern = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto first = manager.create(pattern.size());
    manager.getBuffer(first).writeData(0, pattern.data(), pattern.size());

    manager.recycle();  // rewinds the arena; the same memory will be handed out again

    const auto reborn = manager.create(pattern.size());
    const auto& buffer = manager.getBuffer(reborn);
    for (size_t i = 0; i < pattern.size(); ++i) {
        EXPECT_EQ(buffer.getData()[i], 0u);  // BufferManager zeroed the reused bytes
    }
}

TEST(VirtualMachineBufferManagerArenaTest, CreateZeroesMemoryReusedFromADeletedBuffer) {
    // BufferManager -- not the allocator -- zero-initializes owned buffers in create(). Deleting the
    // top-of-stack buffer lets the arena hand its (dirtied) slot straight back, so the next create() must
    // clear the previous buffer's bytes even without a recycle().
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, makeCountingArena(counters));

    const std::array<uint8_t, 8> pattern = {1, 2, 3, 4, 5, 6, 7, 8};
    const auto first = manager.create(pattern.size());
    auto& firstBuffer = manager.getBuffer(first);
    firstBuffer.writeData(0, pattern.data(), pattern.size());
    const auto* firstData = firstBuffer.getData();
    manager.deleteBuffer(first);  // LIFO pop: the arena will reuse this exact slot

    const auto reborn = manager.create(pattern.size());
    const auto& buffer = manager.getBuffer(reborn);
    ASSERT_EQ(buffer.getData(), firstData);  // same slot handed back
    for (size_t i = 0; i < pattern.size(); ++i) {
        EXPECT_EQ(buffer.getData()[i], 0u);  // create() cleared the slot the deleted buffer dirtied
    }
}

// ============================================================
// BufferManager over a bare leaf allocator
//
// BufferManager must not depend on wrapping an arena: it frees its own Owned buffers in recycle(),
// move assignment and destruction. These tests erase a bare CountingLeaf and assert every teardown path
// balances allocate()/deallocate(), so leaf-owned memory is reclaimed and never double-freed.
// ============================================================

TEST(VirtualMachineBufferManagerLeafTest, RecycleDeallocatesOwnedBuffersWithLeafAllocator) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&counters}});

    manager.create(/*size=*/64);
    manager.create(/*size=*/128);
    manager.create(/*size=*/256);
    ASSERT_EQ(counters.live, 3);

    manager.recycle();
    EXPECT_EQ(counters.live, 0);  // every owned block freed back to the leaf
    EXPECT_EQ(manager.getCurrentMemorySize(), 0u);

    // The manager stays usable across rounds and never leaks.
    for (int round = 0; round < 5; ++round) {
        manager.create(/*size=*/64);
        manager.create(/*size=*/64);
        manager.recycle();
        EXPECT_EQ(counters.live, 0);
    }
}

TEST(VirtualMachineBufferManagerLeafTest, RecycleDeallocatesOwnedBuffersWithLeafAllocator2) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&counters}});

    manager.create(/*size=*/64);
    manager.create(/*size=*/128);
    ASSERT_EQ(counters.live, 2);

    manager.recycle();
    EXPECT_EQ(counters.live, 0);
    EXPECT_EQ(manager.getCurrentMemorySize(), 0u);
}

TEST(VirtualMachineBufferManagerLeafTest, DestructorDeallocatesOwnedBuffersWithLeafAllocator) {
    AllocCounters counters;
    {
        BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&counters}});
        manager.create(/*size=*/64);
        manager.create(/*size=*/128);
        ASSERT_EQ(counters.live, 2);
    }
    EXPECT_EQ(counters.live, 0);  // destruction frees the leaf-owned blocks
}

TEST(VirtualMachineBufferManagerLeafTest, MoveAssignDeallocatesReplacedOwnedBuffersWithLeafAllocator) {
    AllocCounters victimCounters;
    AllocCounters freshCounters;

    BufferManager victim(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&victimCounters}});
    victim.create(/*size=*/64);
    victim.create(/*size=*/128);
    ASSERT_EQ(victimCounters.live, 2);

    // Replacing the populated manager must free the blocks it still owns (mirrors ensureBufferManager's
    // _bufferManager = BufferManager(...) on context change).
    victim = BufferManager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&freshCounters}});
    EXPECT_EQ(victimCounters.live, 0);  // the replaced manager's owned blocks were freed

    // The moved-in manager remains usable and leak-free.
    victim.create(/*size=*/32);
    EXPECT_EQ(freshCounters.live, 1);
    victim.recycle();
    EXPECT_EQ(freshCounters.live, 0);
}

TEST(VirtualMachineBufferManagerLeafTest, UnownedBuffersAreNotDeallocatedByRecycle) {
    AllocCounters counters;
    BufferManager manager(DEFAULT_BUFFER_MEMORY_LIMIT_BYTES, AnyAllocator{CountingLeaf{&counters}});

    std::array<uint8_t, 16> external = {};
    const auto owned = manager.create(/*size=*/64);
    manager.createFromMemory(external.data(), external.size(), Permission::Read);  // unowned: external memory
    manager.createFromBuffer(owned, /*offset=*/0, /*size=*/32);                    // unowned: derived view
    ASSERT_EQ(counters.live, 1);  // only the single owned buffer hit the leaf

    manager.recycle();
    EXPECT_EQ(counters.live, 0);  // exactly one deallocate matching the one allocate, no double free
}

// ============================================================
// Stress Tests
// ============================================================

TEST(VirtualMachineBufferManagerStressTest, CreateAndDeleteManyBuffers) {
    constexpr size_t count = 10'000;
    constexpr size_t bufferSize = 1;
    BufferManager manager(count * bufferSize, makeHostAllocator());

    for (size_t i = 0; i < count; ++i) {
        const auto handle = manager.create(bufferSize);
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}

TEST(VirtualMachineBufferManagerStressTest, CreateAndDeleteManyExternalBuffers) {
    constexpr size_t iterations = 10'000;
    BufferManager manager(0, makeHostAllocator());  // No owned memory limit needed
    std::array<uint8_t, 1> dummy = {};

    for (size_t i = 0; i < iterations; ++i) {
        const auto handle = manager.createFromMemory(dummy.data(), dummy.size(), Permission::Read);
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}

TEST(VirtualMachineBufferManagerStressTest, CreateManyExternalBuffersThenDeleteAll) {
    constexpr size_t count = 10'000;
    BufferManager manager(0, makeHostAllocator());
    std::array<uint8_t, 1> dummy = {};

    std::vector<BufferHandle> handles;
    handles.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        handles.push_back(manager.createFromMemory(dummy.data(), dummy.size(), Permission::Read));
    }
    for (const auto& handle : handles) {
        ASSERT_TRUE(manager.exists(handle));
        manager.deleteBuffer(handle);
        ASSERT_FALSE(manager.exists(handle));
    }
}
