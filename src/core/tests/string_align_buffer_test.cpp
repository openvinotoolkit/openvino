// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

namespace ov {
namespace test {

using testing::HasSubstr;

using StringAlignedBufferTest = testing::Test;

TEST_F(StringAlignedBufferTest, default_ctor) {
    StringAlignedBuffer buffer;

    ASSERT_EQ(buffer.get_ptr(), nullptr);
    EXPECT_EQ(buffer.get_num_elements(), 0);
    EXPECT_EQ(buffer.size(), 0);
}

TEST_F(StringAlignedBufferTest, create_not_initialized_but_init_before_destruction) {
    constexpr size_t exp_size = 4 * sizeof(std::string) + 1;
    StringAlignedBuffer buffer{4, exp_size, 64, false};

    ASSERT_NE(buffer.get_ptr(), nullptr);
    EXPECT_EQ(buffer.get_num_elements(), 4);
    EXPECT_EQ(buffer.size(), exp_size);

    // uninitialized buffer must be initialized by user before dtor call to avoid segfault
    std::uninitialized_fill_n(buffer.get_ptr<std::string>(), buffer.get_num_elements(), std::string{});
}

TEST_F(StringAlignedBufferTest, create_initialized) {
    constexpr size_t exp_size = sizeof(std::string) * 5;

    StringAlignedBuffer buffer(5, exp_size, 8, true);

    ASSERT_NE(buffer.get_ptr(), nullptr);
    EXPECT_EQ(buffer.get_num_elements(), 5);
    EXPECT_EQ(buffer.size(), exp_size);
}

TEST_F(StringAlignedBufferTest, create_initialized_not_enough_space) {
    constexpr size_t exp_size = sizeof(std::string) * 5;

    OV_EXPECT_THROW(StringAlignedBuffer buffer(5, exp_size - 1, 8, true),
                    AssertFailure,
                    HasSubstr("is not enough to store 5 std::string objects"));
}

TEST_F(StringAlignedBufferTest, create_not_initialized_not_enough_space) {
    constexpr size_t exp_size = sizeof(std::string) * 5;

    OV_EXPECT_THROW(StringAlignedBuffer buffer(5, exp_size - 1, 8, false),
                    AssertFailure,
                    HasSubstr("is not enough to store 5 std::string objects"));
}

class SharedStringAlignedBufferTest : public testing::Test {
protected:
    const size_t exp_size = 11 * sizeof(std::string);
    const std::string msg_at_0 = "test input message at index 0";

    StringAlignedBuffer buffer{11, exp_size, 0, true};
};

TEST_F(SharedStringAlignedBufferTest, dtor_not_destroy_input_buffer) {
    *buffer.get_ptr<std::string>() = msg_at_0;

    {
        SharedStringAlignedBuffer shared_buffer(buffer.get_ptr<char>(), buffer.size());

        ASSERT_EQ(shared_buffer.get_ptr(), buffer.get_ptr());
        EXPECT_EQ(shared_buffer.get_num_elements(), buffer.get_num_elements());
        EXPECT_EQ(shared_buffer.size(), exp_size);
        EXPECT_EQ(shared_buffer.get_ptr<const std::string>()[0], msg_at_0);
    }

    ASSERT_NE(buffer.get_ptr(), nullptr);
    EXPECT_EQ(buffer.get_num_elements(), 11);
    EXPECT_EQ(buffer.size(), exp_size);
    EXPECT_EQ(buffer.get_ptr<const std::string>()[0], msg_at_0);
}

TEST_F(SharedStringAlignedBufferTest, create_with_smaller_size_than_input_buffer) {
    buffer.get_ptr<std::string>()[0] = msg_at_0;

    const SharedStringAlignedBuffer shared_buffer{buffer.get_ptr<char>(), sizeof(std::string)};

    ASSERT_EQ(shared_buffer.get_ptr(), buffer.get_ptr());
    EXPECT_EQ(shared_buffer.get_num_elements(), 1);
    EXPECT_EQ(shared_buffer.size(), sizeof(std::string));
    EXPECT_EQ(shared_buffer.get_ptr<std::string>()[0], msg_at_0);

    ASSERT_NE(buffer.get_ptr(), nullptr);
    EXPECT_EQ(*buffer.get_ptr<const std::string>(), msg_at_0);
}

}  // namespace test
}  // namespace ov
