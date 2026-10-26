//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Device-backed tests for the bytecode VM's Level Zero allocator. Unlike the composable-allocator unit
// tests (which run over host memory), these exercise the real zeMemAllocHost/zeMemFree leaf against a live
// Level Zero context, so they only run where an NPU is present and GTEST_SKIP otherwise.

#include "level_zero_allocator.hpp"
#include "allocator_core.hpp"
#include "common/npu_test_env_cfg.hpp"

#include <common_test_utils/test_common.hpp>
#include <functional_test_utils/skip_tests_config.hpp>
#include <intel_npu/utils/utils.hpp>
#include <intel_npu/utils/zero/zero_init.hpp>

#include <ze_api.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

using intel_npu::vm::AnyAllocator;
using intel_npu::vm::Block;
using intel_npu::vm::LevelZeroAllocator;
using intel_npu::vm::makeLevelZeroAllocator;

using LevelZeroAllocatorDeviceTestParams = std::string;

namespace {

constexpr size_t kPage = intel_npu::utils::STANDARD_PAGE_SIZE;

bool isPageAligned(const uint8_t* ptr) {
    return reinterpret_cast<uintptr_t>(ptr) % kPage == 0;
}

}  // namespace

class LevelZeroAllocatorDeviceTest :
        public testing::WithParamInterface<LevelZeroAllocatorDeviceTestParams>,
        public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<LevelZeroAllocatorDeviceTestParams>& obj) {
        std::string targetDevice = obj.param;

        std::ostringstream result;
        result << "target_platform=" << ov::test::utils::getTestsPlatformFromEnvironmentOr(targetDevice) << "_";
        result << "target_device=" << targetDevice << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        _targetDevice = GetParam();
        try {
            _initStruct = std::make_shared<intel_npu::ZeroInitStructsHolder>();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "No Level Zero / NPU device available: " << e.what();
        }
        _context = _initStruct->getContext();
        ASSERT_NE(_context, nullptr);
    }

    std::string _targetDevice;
    std::shared_ptr<intel_npu::ZeroInitStructsHolder> _initStruct;
    ze_context_handle_t _context = nullptr;
};

// The bare leaf: zeMemAllocHost-backed, page-aligned, host-visible, reusable across allocate/deallocate.
TEST_P(LevelZeroAllocatorDeviceTest, LeafAllocatesPageAlignedHostVisibleMemory) {
    LevelZeroAllocator allocator(_context);

    const Block block = allocator.allocate(4 * kPage);
    ASSERT_TRUE(block);
    EXPECT_TRUE(isPageAligned(block.ptr));
    EXPECT_EQ(block.size, 4u * kPage);

    // Host-visible: the CPU can write and read the device buffer back.
    for (size_t i = 0; i < block.size; ++i) {
        block.ptr[i] = static_cast<uint8_t>(i);
    }
    for (size_t i = 0; i < block.size; ++i) {
        ASSERT_EQ(block.ptr[i], static_cast<uint8_t>(i));
    }

    allocator.deallocate(block);

    // A fresh allocation after a free still succeeds.
    const Block again = allocator.allocate(kPage);
    ASSERT_TRUE(again);
    EXPECT_TRUE(isPageAligned(again.ptr));
    allocator.deallocate(again);
}

// The Level Zero device allocator (GrowingArena<LevelZeroAllocator, BumpRegion<page>, ...>): page-aligned, distinct,
// host-visible sub-buffers; recycle() reuses the same memory; release() frees and the arena re-grows.
TEST_P(LevelZeroAllocatorDeviceTest, DefaultAllocatorHandsOutAlignedDistinctReusableBuffers) {
    AnyAllocator allocator = makeLevelZeroAllocator(_context);
    ASSERT_TRUE(allocator);

    const Block a = allocator.allocate(1024);
    const Block b = allocator.allocate(2048);
    const Block c = allocator.allocate(4096);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    ASSERT_TRUE(c);

    for (const Block& block : {a, b, c}) {
        EXPECT_TRUE(isPageAligned(block.ptr));
    }
    EXPECT_NE(a.ptr, b.ptr);
    EXPECT_NE(a.ptr, c.ptr);
    EXPECT_NE(b.ptr, c.ptr);

    // Host-visible: writing through the returned pointers must not fault.
    a.ptr[0] = 0xAB;
    c.ptr[c.size - 1] = 0xCD;

    // recycle() rewinds the arena: replaying the same requests reuses the same addresses (no growth).
    allocator.recycle();
    const Block a2 = allocator.allocate(1024);
    EXPECT_EQ(a2.ptr, a.ptr);

    // release() frees the slabs back to the device; the allocator stays usable and re-grows on demand.
    allocator.release();
    const Block afterRelease = allocator.allocate(1024);
    ASSERT_TRUE(afterRelease);
    EXPECT_TRUE(isPageAligned(afterRelease.ptr));
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest_level_zero_allocator, LevelZeroAllocatorDeviceTest,
                         ::testing::Values(test_utils::TARGET_DEVICE), LevelZeroAllocatorDeviceTest::getTestCaseName);
