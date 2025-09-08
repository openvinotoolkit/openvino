// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/aligned_buffer.hpp"

#include "gtest/gtest.h"

using namespace ov;

TEST(aligned_buffer, alignment) {
    AlignedBuffer buffer(100, 64);
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
