// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/util/reservable_buffer.hpp"

using namespace testing;

namespace ov::test {

TEST(ReservableBufferTest, construction) {
    OV_EXPECT_THROW(util::ReservableBuffer{0}, std::runtime_error, HasSubstr("Zero size buffer makes no sense"));

    EXPECT_NO_THROW(util::ReservableBuffer{4096});
}

TEST(ReservableBufferTest, stable_pointer) {
    util::ReservableBuffer buf{4096};
    EXPECT_TRUE(buf.last_error().empty());

    const auto ptr = buf.pointer();
    ASSERT_NE(ptr, nullptr);
    EXPECT_TRUE(buf.last_error().empty());

    ASSERT_TRUE(buf.acquire());
    EXPECT_TRUE(buf.last_error().empty());
    EXPECT_EQ(buf.pointer(), ptr);

    ASSERT_NO_THROW(buf.evict());
    EXPECT_TRUE(buf.last_error().empty());
    EXPECT_EQ(buf.pointer(), ptr);

    ASSERT_TRUE(buf.acquire());
    EXPECT_TRUE(buf.last_error().empty());
    EXPECT_EQ(buf.pointer(), ptr);
}

TEST(ReservableBufferTest, write) {
    util::ReservableBuffer buf{257};
    ASSERT_TRUE(buf.acquire());

    const auto ptr = static_cast<uint8_t*>(buf.pointer());
    const auto sz = buf.size();
    ASSERT_NE(ptr, nullptr);
    ASSERT_EQ(sz, 257);

    std::vector<uint8_t> test_data(sz, 0x55);
    ASSERT_NO_THROW(memcpy(ptr, test_data.data(), sz));
    EXPECT_THAT(test_data, ElementsAreArray(ptr, sz));
}
}  // namespace ov::test
