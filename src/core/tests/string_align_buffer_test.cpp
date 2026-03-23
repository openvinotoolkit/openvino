// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"

namespace ov::test {

using testing::HasSubstr;

using StringAlignedBufferTest = testing::Test;

using header_element_t = int32_t;  // Type of a single element in the header (strings_count and offsets)

namespace {

/// @brief Creates a packed string tensor byte stream from given header and strings data.
/// @param header The header elements of the string tensor.
/// @param strings The string data of the tensor.
/// @return A vector of bytes representing the packed string tensor.
std::vector<uint8_t> pack_string_tensor(const std::vector<header_element_t>& header,
                                        const std::vector<uint8_t>& strings = {}) {
    std::vector<uint8_t> tensor;
    tensor.reserve(header.size() * sizeof(header_element_t) + strings.size());
    for (const auto value : header) {
        const uint8_t* characters = reinterpret_cast<const uint8_t*>(&value);
        tensor.insert(tensor.end(), characters, characters + sizeof(header_element_t));
    }
    tensor.insert(tensor.end(), strings.begin(), strings.end());
    return tensor;
}

/// @brief Introduces invalid elements count in the header of the packed string tensor to trigger error handling in
/// unpacking function.
/// @param tensor The packed string tensor.
/// @param elements_count The invalid elements count to be set in the header.
void tamper_with_elements_count(std::vector<uint8_t>& tensor, const header_element_t elements_count) {
    constexpr size_t elements_count_index = 0;
    const uint8_t* characters = reinterpret_cast<const uint8_t*>(&elements_count);
    std::copy(characters, characters + sizeof(header_element_t), tensor.begin() + elements_count_index);
}

}  // namespace

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

/// @brief Test case for zero number of strings in packed string tensor.
/// Expecting an empty buffer.
/// num_strings = 0
TEST(StringUnpackTensorTest, ZeroNumberOfStringsYieldsEmptyBuffer) {
    constexpr auto strings_count = 0;

    const auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count});
    const auto result = AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
        reinterpret_cast<const char*>(tensor.data()),
        tensor.size());

    EXPECT_EQ(result->get_num_elements(), 0);
}

/// @brief Test case for missing number of strings in packed string tensor.
/// Expecting AssertFailure with message about missing strings count.
/// num_strings = <missing>
TEST(StringUnpackTensorTest, MissingNumberOfStringsFails) {
    const std::vector<uint8_t> strings = {'0', '1'};

    const auto tensor = pack_string_tensor(std::vector<header_element_t>{}, strings);
    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("no strings count in the packed string tensor"));
}

/// @brief Test case for negative number of strings in packed string tensor.
/// Expecting AssertFailure with message about negative number of strings.
/// num_strings = -1, which is invalid because number of strings cannot be negative.
TEST(StringUnpackTensorTest, NegativeNumberOfStringsFails) {
    constexpr auto strings_count = -1;

    const auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count});
    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("negative number of strings"));
}

/// @brief Test case for header size exceeding indexable range.
/// Expecting AssertFailure with message about header size calculation overflow.
/// num_strings = <plenty>
TEST(StringUnpackTensorTest, HeaderSizeOverflowFails) {
    constexpr auto strings_count = 1;
    constexpr auto strings_count_tampered = std::numeric_limits<header_element_t>::max();

    if (strings_count_tampered < std::numeric_limits<size_t>::max() / sizeof(header_element_t)) {
        GTEST_SKIP() << "Header size calculation does not overflow on this platform";
    }

    const std::vector<uint8_t> strings = {'0', '1', '2', '3', '4'};
    auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count, 0, 5}, strings);

    tamper_with_elements_count(tensor, strings_count_tampered);

    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("header size overflow detected"));
}

/// @brief Test case for header size exceeding buffer bounds in packed string tensor.
/// Expecting AssertFailure with message about header exceeds provided buffer size.
/// num_strings = 10, header: [10, 0, end0=3, end1=5], but buffer too small
TEST(StringUnpackTensorTest, HeaderSizeBeyondBufferFails) {
    constexpr auto strings_count = 2;
    constexpr auto strings_count_tampered = 10;

    const std::vector<uint8_t> strings = {'0', '1', '2', '3', '4'};
    auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count, 0, 3, 5}, strings);

    tamper_with_elements_count(tensor, strings_count_tampered);

    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("header exceeds provided buffer size"));
}

/// @brief Test case for negative offsets in packed string tensor.
/// Expecting AssertFailure with message about begin offset greater than end offset.
/// num_strings = 2, header: [2, 0, end0=-3, end1=5]
TEST(StringUnpackTensorTest, NegativeOffsetsFails) {
    constexpr auto strings_count = 2;

    const std::vector<uint8_t> strings = {'0', '1', '2', '3', '4'};
    const auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count, 0, -3, 5}, strings);
    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("negative string offset in the packed string tensor"));
}

/// @brief Test case for decreasing offsets in packed string tensor.
/// Expecting AssertFailure with message about begin offset greater than end offset.
/// num_strings = 2, header: [2, 0, end0=5, end1=3]
TEST(StringUnpackTensorTest, DecreasingOffsetsFails) {
    constexpr auto strings_count = 2;

    const std::vector<uint8_t> strings = {'0', '1', '2', '3', '4'};
    const auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count, 0, 5, 3}, strings);
    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("begin offset greater than end offset"));
}

/// @brief Test case for string offset exceeding buffer bounds in packed string tensor.
/// Expecting AssertFailure with message about string offset exceeds buffer bounds.
/// num_strings = 2, header: [2, 0, end0=10, end1=20], but buffer too small
TEST(StringUnpackTensorTest, OffsetBeyondBufferFails) {
    constexpr auto strings_count = 2;

    const std::vector<uint8_t> strings = {'0', '1', '2', '3', '4'};
    const auto tensor = pack_string_tensor(std::vector<header_element_t>{strings_count, 0, 10, 20}, strings);
    OV_EXPECT_THROW(AttributeAdapter<std::shared_ptr<StringAlignedBuffer>>::unpack_string_tensor(
                        reinterpret_cast<const char*>(tensor.data()),
                        tensor.size()),
                    AssertFailure,
                    HasSubstr("string offset exceeds buffer bounds"));
}

}  // namespace ov::test
