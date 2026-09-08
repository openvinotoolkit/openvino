// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/aligned_buffer.hpp"

#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

namespace ov::test {

TEST(aligned_buffer, alloc_failure_throws_bad_alloc) {
    // A request this large cannot be satisfied by aligned_alloc but does not overflow size_t,
    // so the allocator returns nullptr instead of throwing. The constructor must convert that
    // into a catchable std::bad_alloc rather than storing a null buffer.
    constexpr size_t huge = std::numeric_limits<size_t>::max() / 2;
    OV_EXPECT_THROW({ ov::AlignedBuffer buffer(huge, 64); }, std::bad_alloc, testing::_);
}

TEST(aligned_buffer, alignment) {
    AlignedBuffer buffer(100, 64);
    ASSERT_NE(buffer.get_ptr(), nullptr);
    size_t addr = reinterpret_cast<size_t>(buffer.get_ptr()) % 64;
    EXPECT_EQ(addr, 0);
}

TEST(aligned_buffer, move) {
    {
        AlignedBuffer buffer1(100, 64);
        AlignedBuffer buffer2(std::move(buffer1));
        EXPECT_EQ(buffer1.size(), 0);
        EXPECT_EQ(buffer1.get_ptr(), nullptr);
        EXPECT_EQ(buffer2.size(), 100);
        EXPECT_NE(buffer2.get_ptr(), nullptr);
    }
    {
        AlignedBuffer buffer1(100, 64);
        AlignedBuffer buffer2;
        buffer2 = std::move(buffer1);
        EXPECT_EQ(buffer1.size(), 0);
        EXPECT_EQ(buffer1.get_ptr(), nullptr);
        EXPECT_EQ(buffer2.size(), 100);
        EXPECT_NE(buffer2.get_ptr(), nullptr);
    }
}
}  // namespace ov::test
