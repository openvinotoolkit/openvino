// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/vcl/vcl_allocator.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"

using namespace intel_npu;

class VclAllocatorUnitTests : public ::testing::Test {
protected:
    void SetUp() override {
        allocator = std::make_shared<vcl_allocator_3>();
    }

    std::shared_ptr<vcl_allocator_3> allocator;
};

TEST_F(VclAllocatorUnitTests, CheckAllocateAndDeallocateTracking) {
    size_t size1 = 100, size2 = 200, size3 = 300;

    // Simulate VCL allocating multiple temporary buffers
    uint8_t* ptr1 = allocator->allocate(allocator.get(), size1);
    uint8_t* ptr2 = allocator->allocate(allocator.get(), size2);
    uint8_t* ptr3 = allocator->allocate(allocator.get(), size3);

    // Verify allocations are correctly retained in the tracking vector
    EXPECT_EQ(allocator->m_info.size(), 3);
    EXPECT_EQ(allocator->m_info[0].first, ptr1);
    EXPECT_EQ(allocator->m_info[1].first, ptr2);
    EXPECT_EQ(allocator->m_info[2].first, ptr3);

    // Simulate VCL internally freeing one of its unused buffers
    allocator->deallocate(allocator.get(), ptr2);

    // Verify it is erased and no longer tracked
    EXPECT_EQ(allocator->m_info.size(), 2);
    EXPECT_EQ(allocator->m_info[0].first, ptr1);
    EXPECT_EQ(allocator->m_info[1].first, ptr3);

    // Simulate returning the 'blob' back to OpenVINO and transferring it to ov::Tensor
    auto it = std::find_if(allocator->m_info.begin(),
                           allocator->m_info.end(),
                           [ptr3](const std::pair<uint8_t*, size_t>& item) {
                               return item.first == ptr3;
                           });
    ASSERT_NE(it, allocator->m_info.end());

    {
        // Transfer blob to ov::Tensor with a custom deleter capturing our allocator
        ov::Tensor tensor = make_tensor_from_aligned_addr(ptr3, it->second, allocator);
        EXPECT_EQ(tensor.data(), ptr3);

        // Emulate the targeted removal logic in compile() avoiding memory leaks
        allocator->m_info.erase(it);

        EXPECT_EQ(allocator->m_info.size(), 1);
        EXPECT_EQ(allocator->m_info[0].first, ptr1);
    }
}

TEST_F(VclAllocatorUnitTests, CheckOrphanMemoryCleanup) {
    size_t size1 = 100;
    // Simulate VCL allocating multiple temporary buffers
    uint8_t* ptr1 = allocator->allocate(allocator.get(), size1);

    EXPECT_EQ(allocator->m_info.size(), 1);
    EXPECT_EQ(allocator->m_info[0].first, ptr1);

    // Simulate VCL crash or memory leak where it doesn't deallocate the buffer
    // and the buffer is not wrapped in an ov::Tensor
    allocator.reset();
}

TEST_F(VclAllocatorUnitTests, CheckWsOneShotMultipleOutputs) {
    size_t size1 = 400, size2 = 500;

    // Simulate compileWsOneShot workflow dealing with multiple outputs naturally generated
    uint8_t* ptr1 = allocator->allocate(allocator.get(), size1);
    uint8_t* ptr2 = allocator->allocate(allocator.get(), size2);

    EXPECT_EQ(allocator->m_info.size(), 2);
    EXPECT_EQ(allocator->m_info[0].first, ptr1);
    EXPECT_EQ(allocator->m_info[1].first, ptr2);

    {
        std::vector<ov::Tensor> tensors;
        for (const auto& blob : allocator->m_info) {
            tensors.emplace_back(make_tensor_from_aligned_addr(blob.first, blob.second, allocator));
        }

        // Emulate clear tracking operation within WS oneshot procedure
        allocator->m_info.clear();

        EXPECT_EQ(allocator->m_info.size(), 0);
        EXPECT_EQ(tensors.size(), 2);
    }
}
