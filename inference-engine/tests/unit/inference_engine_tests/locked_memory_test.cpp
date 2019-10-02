// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_common.hpp"
#include "mock_allocator.hpp"

using namespace std;

class LockedMemoryTest : public TestsCommon {
protected:
    unique_ptr<MockAllocator> createMockAllocator() {
        return unique_ptr<MockAllocator>(new MockAllocator());
    }
};

using namespace InferenceEngine;
using namespace ::testing;


TEST_F(LockedMemoryTest, canUnlockMemoryAfterUsage) {

    auto allocator = createMockAllocator();

    char array [] = {1,2,3};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 1);
        //force locking of memory
        auto UNUSED t = x[0];
    }
}


TEST_F(LockedMemoryTest, canReadFromLockedMemory) {

    auto allocator = createMockAllocator();

    char array [] = {1,2,3,4,5};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 0);
        //we are getting first element
        ASSERT_EQ(1, x[0]);
    }
}


TEST_F(LockedMemoryTest, canWriteToLockedMemory) {

    auto allocator = createMockAllocator();

    char array [] = {1,2,3,4,5};

    EXPECT_CALL(*allocator.get(), lock((void*)1, _)).WillRepeatedly(Return((void*)array));
    EXPECT_CALL(*allocator.get(), unlock(_)).Times(1);
    {
        auto x = LockedMemory<char>(allocator.get(), (void *) 1, 0);

        //we are getting first element
        ASSERT_EQ(std::distance(array, &x[0]), 0);
        x[0] = 5;
    }
    EXPECT_EQ(array[0], 5);

}
