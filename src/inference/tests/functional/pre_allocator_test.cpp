// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <vector>

#include "details/ie_pre_allocator.hpp"
#include "ie_allocator.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

class PreallocatorTests : public ::testing::Test {
protected:
    std::vector<float> mybuf;

    void SetUp() override {
        mybuf.resize(10);
        allocator = details::make_pre_allocator(&*mybuf.begin(), mybuf.size());
    }
    std::shared_ptr<IAllocator> allocator;
};

TEST_F(PreallocatorTests, canAccessPreAllocatedMemory) {
    void* handle = allocator->alloc(3);
    float* ptr = reinterpret_cast<float*>(allocator->lock(handle));

    mybuf = {1.1f, 2.2f, 3.3f};

    ASSERT_EQ(ptr, &*mybuf.begin());
    ASSERT_EQ(ptr[0], 1.1f);
    ASSERT_EQ(ptr[1], 2.2f);
    ASSERT_EQ(ptr[2], 3.3f);
}

TEST_F(PreallocatorTests, canNotAllocateMoreMemory) {
    // large block such as 10k will result in nullptr
    EXPECT_EQ(nullptr, allocator->lock(allocator->alloc(10 * sizeof(float) + 1)));
    EXPECT_NE(nullptr, allocator->lock(allocator->alloc(10 * sizeof(float))));
}

TEST_F(PreallocatorTests, canNotLockWrongHandle) {
    void* handle = allocator->alloc(3);
    EXPECT_EQ(nullptr, allocator->lock(1 + reinterpret_cast<int*>(handle)));
}
