// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>

#include "ie_allocator.hpp"

using namespace ::testing;
using namespace std;
using namespace InferenceEngine;

class SystemAllocatorTests: public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
        allocator = details::shared_from_irelease(CreateDefaultAllocator());
    }
    std::shared_ptr<IAllocator> allocator;
public:

};

TEST_F(SystemAllocatorTests, canAllocate) {
    void* handle = allocator->alloc(100);
    EXPECT_NE(nullptr, handle);
    allocator->free(handle);
}

TEST_F(SystemAllocatorTests, canLockAllocatedMemory) {
    //large block such as 10k will result in sigsegv if not allocated
    void * handle  = allocator->alloc(10000);
    char * ptr = (char *)allocator->lock(handle);
    ptr [9999] = 11;
    ASSERT_EQ(ptr[9999], 11);
    allocator->unlock(ptr);
    allocator->free(handle);
}
