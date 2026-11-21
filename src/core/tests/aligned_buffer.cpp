// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/aligned_buffer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "common_test_utils/test_assertions.hpp"
#include "gtest/gtest.h"

namespace ov::test {

TEST(aligned_buffer, alloc_failure_throws) {
    constexpr size_t huge = std::numeric_limits<size_t>::max() / 2;
    OV_EXPECT_THROW(
        { ov::AlignedBuffer buffer(huge, 64); },
        ov::AssertFailure,
        testing::HasSubstr("Failed to allocate"));
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

TEST(aligned_buffer, mutable_buffer_data_matches_get_ptr) {
    AlignedBuffer buffer(100, 64);
    auto&& const_buffer = std::as_const(buffer);

    EXPECT_EQ(buffer.data(), buffer.get_ptr());
    EXPECT_EQ(const_buffer.data(), const_buffer.get_ptr());
    EXPECT_EQ(buffer.data_as<char>(), buffer.get_ptr<char>());
    EXPECT_EQ(buffer.size(), 100);
}

TEST(aligned_buffer, mutable_buffer_interface) {
    AlignedBuffer buffer(16, 64);
    std::fill_n(buffer.data_as<uint8_t>(), buffer.size(), uint8_t{0x5A});

    EXPECT_EQ(buffer.size(), 16);
    EXPECT_EQ(buffer.get_properties().offset, 0);
    EXPECT_FALSE(buffer.get_id().has_value());

    const auto view = buffer.view();
    EXPECT_EQ(view.size(), 16);
    EXPECT_TRUE(std::all_of(view.begin(), view.end(), [](auto&& byte) {
        return byte == std::byte{0x5A};
    }));
}

TEST(aligned_buffer, mutable_buffer_default_is_empty) {
    AlignedBuffer buffer;
    EXPECT_EQ(buffer.data(), nullptr);
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.get_properties().offset, 0);
}

TEST(aligned_buffer, hint_prefetch_returns_no_pending_token) {
    AlignedBuffer buffer(16, 64);
    EXPECT_FALSE(buffer.hint_prefetch().valid());
    EXPECT_NO_THROW(buffer.hint_evict());
}
}  // namespace ov::test
