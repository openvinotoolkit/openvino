// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory.hpp"

#include <common_test_utils/ov_tensor_utils.hpp>

#include "intel_npu/utils/zero/zero_utils.hpp"
#include "openvino/runtime/make_tensor.hpp"

using namespace intel_npu;

void MemoryAllocator::SetUp() {
    int allocatorFlag = GetParam();
    allocator = std::make_shared<zeroMemory::HostMemAllocator>(ZeroInitStructsHolder::getInstance(), allocatorFlag);
}

void MemoryAllocator::TearDown() {}

TEST_P(MemoryAllocator, AllocateTwice) {
    ov::Shape shape = {1, 1, 128};
    auto byteSize = ov::shape_size(shape) * ov::element::f32.size();
    float* data = static_cast<float*>(::operator new(byteSize, std::align_val_t(4096)));
    std::shared_ptr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, shape, data);
    ::operator delete(data, std::align_val_t(4096));

    sharedAllocator =
        std::make_shared<zeroMemory::HostMemSharedAllocator>(ZeroInitStructsHolder::getInstance(), tensor, GetParam());

    size_t size = 1 << 10;
    void* ptr = sharedAllocator->allocate(size);
    EXPECT_NE(ptr, nullptr);

    ptr = sharedAllocator->allocate(size);
    EXPECT_NE(ptr, nullptr);

    bool result =
        zeroUtils::memory_was_allocated_in_the_same_l0_context(ZeroInitStructsHolder::getInstance()->getContext(), ptr);
    ASSERT_EQ(result, true);
}

TEST_P(MemoryAllocator, AllocateAboveMax) {
    void* ptr = allocator->allocate(161'061'273'608);
    ASSERT_EQ(ptr, nullptr);

    bool result =
        zeroUtils::memory_was_allocated_in_the_same_l0_context(ZeroInitStructsHolder::getInstance()->getContext(), ptr);
    ASSERT_EQ(result, false);
}

TEST_P(MemoryAllocator, DeallocateNullHandle) {
    void* ptr = nullptr;
    bool result = allocator->deallocate(ptr, 0xDEADBEEF);
    ASSERT_EQ(result, false);

    result =
        zeroUtils::memory_was_allocated_in_the_same_l0_context(ZeroInitStructsHolder::getInstance()->getContext(), ptr);
    ASSERT_EQ(result, false);
}

TEST_P(MemoryAllocator, AllocateThenDeallocate) {
    void* ptr;
    size_t size = 1 << 10;

    ptr = allocator->allocate(size);
    ASSERT_NE(ptr, nullptr);

    bool result = allocator->deallocate(ptr, size);
    ASSERT_EQ(result, true);

    result =
        zeroUtils::memory_was_allocated_in_the_same_l0_context(ZeroInitStructsHolder::getInstance()->getContext(), ptr);
    ASSERT_EQ(result, false);
}

TEST_P(MemoryAllocator, DeallocateUnknownHandle) {
    size_t size = 1 << 10;
    void* ptr = ::operator new(size, std::align_val_t(4096));

    bool result = allocator->deallocate(ptr, size);
    ASSERT_EQ(result, true);

    result =
        zeroUtils::memory_was_allocated_in_the_same_l0_context(ZeroInitStructsHolder::getInstance()->getContext(), ptr);
    ASSERT_EQ(result, true);

    ::operator delete(ptr, std::align_val_t(4096));
}

std::vector<int> allocatorFlags = {ZE_HOST_MEM_ALLOC_FLAG_BIAS_CACHED,
                                   ZE_HOST_MEM_ALLOC_FLAG_BIAS_UNCACHED,
                                   ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED,
                                   ZE_HOST_MEM_ALLOC_FLAG_BIAS_INITIAL_PLACEMENT};

INSTANTIATE_TEST_SUITE_P(something,
                         MemoryAllocator,
                         ::testing::ValuesIn(allocatorFlags),
                         MemoryAllocator::getTestCaseName);
