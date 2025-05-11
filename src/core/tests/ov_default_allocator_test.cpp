// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/allocator.hpp"

using OVDefaultAllocatorTest = ::testing::Test;

TEST_F(OVDefaultAllocatorTest, notThrowOnZeroSize) {
    ov::Allocator allocator;
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocator.allocate(0));
    OV_ASSERT_NO_THROW(allocator.deallocate(ptr));
}

TEST_F(OVDefaultAllocatorTest, canAllocateAndDeallocate) {
    ov::Allocator allocator;
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocator.allocate(64));
    OV_ASSERT_NO_THROW(allocator.deallocate(ptr));
}

TEST_F(OVDefaultAllocatorTest, alignedAllocationNotThrow) {
    ov::Allocator allocator;
    OV_ASSERT_NO_THROW(allocator.allocate(64, 64));
}

TEST_F(OVDefaultAllocatorTest, sizedAndAlignedDeallocationNotThrow) {
    ov::Allocator allocator;
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocator.allocate(64));
    OV_ASSERT_NO_THROW(allocator.deallocate(ptr, 64));
    OV_ASSERT_NO_THROW(ptr = allocator.allocate(64, 64));
    OV_ASSERT_NO_THROW(allocator.deallocate(ptr, 64, 64));
}

TEST_F(OVDefaultAllocatorTest, defaultAllocatorsAreEqual) {
    ov::Allocator allocator0, allocator1;
    ASSERT_TRUE(allocator0 == allocator1);
    void* ptr = nullptr;
    OV_ASSERT_NO_THROW(ptr = allocator0.allocate(64));
    OV_ASSERT_NO_THROW(allocator1.deallocate(ptr));
    OV_ASSERT_NO_THROW(ptr = allocator1.allocate(64));
    OV_ASSERT_NO_THROW(allocator0.deallocate(ptr));
}

TEST_F(OVDefaultAllocatorTest, canAllocate10KMemory) {
    ov::Allocator allocator;
    // large block such as 10k will result in sigsegv if not allocated
    void* handle = allocator.allocate(10000);
    char* ptr = reinterpret_cast<char*>(handle);
    ptr[9999] = 11;
    EXPECT_EQ(ptr[9999], 11);
    allocator.deallocate(handle);
}
