// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

void MemoryAllocator::SetUp() {
    ov::Shape shape = {1, 1, 128};
    std::vector<int8_t> input(100, 100);
    std::shared_ptr<ov::ITensor> tensor = ov::make_tensor(ov::element::f32, shape, input.data());

    this->allocator = std::make_shared<zeroMemory::HostMemSharedAllocator>(ZeroInitStructsHolder::getInstance(), tensor, ZE_HOST_MEM_ALLOC_FLAG_BIAS_WRITE_COMBINED);
}

void MemoryAllocator::TearDown() {

}

TEST_P(MemoryAllocator, AllocateTwice) {
    void* ptr;
    size_t size = this->GetParam();
    OV_ASSERT_NO_THROW(ptr = allocator->allocate(size));
    ptr = allocator->allocate(size);
    (void)ptr;
}

std::vector<size_t> aa = {10, 10000, 100000};

INSTANTIATE_TEST_SUITE_P(whatever, MemoryAllocator, ::testing::ValuesIn(aa), MemoryAllocator::getTestCaseName);
