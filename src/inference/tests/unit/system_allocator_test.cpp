// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"

#include "system_allocator.hpp"

using namespace InferenceEngine;
class SystemAllocatorReleaseTests : public CommonTestUtils::TestsCommon {
};

class SystemAllocatorTests : public CommonTestUtils::TestsCommon {
protected:
    void SetUp() override {
        CommonTestUtils::TestsCommon::SetUp();
        ASSERT_EQ(allocator.get(), nullptr);
        allocator = createSystemMemoryAllocator();
        ASSERT_NE(allocator.get(), nullptr);
    }

    void TearDown() override {
        CommonTestUtils::TestsCommon::TearDown();
        ASSERT_NE(allocator.get(), nullptr);
        allocator.reset();
        ASSERT_EQ(allocator.get(), nullptr);
    }

    std::unique_ptr<SystemMemoryAllocator> createSystemMemoryAllocator() {
        return std::unique_ptr<SystemMemoryAllocator>(new SystemMemoryAllocator());
    }

    std::unique_ptr<SystemMemoryAllocator> allocator;
public:
};

TEST_F(SystemAllocatorTests, canAllocate) {
    void *handle0 = allocator->alloc(0);
    void *handle1 = allocator->alloc(100);
    EXPECT_NE(handle0, nullptr);
    EXPECT_NE(handle1, nullptr);
    delete[] reinterpret_cast<char *>(handle0);
    delete[] reinterpret_cast<char *>(handle1);
}

TEST_F(SystemAllocatorTests, canFree) {
    EXPECT_TRUE(allocator->free(nullptr));
    void *handle0 = reinterpret_cast<void *>(new char[0]);
    void *handle1 = reinterpret_cast<void *>(new char[100]);
    EXPECT_TRUE(allocator->free(handle0));
    EXPECT_TRUE(allocator->free(handle1));
}

TEST_F(SystemAllocatorTests, canLockAndUnlockAllocatedMemory) {
    // large block such as 10k will result in sigsegv if not allocated
    void *handle = allocator->alloc(10000);
    char *ptr = reinterpret_cast<char *>(allocator->lock(handle));
    ptr[9999] = 11;
    EXPECT_EQ(ptr[9999], 11);
    allocator->unlock(ptr);
    allocator->free(handle);
}