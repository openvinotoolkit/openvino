// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"

#include <sstream>

#include "gtest/gtest.h"

using ov::SharedStreamBuffer;

TEST(shared_stream_buffer, basic_read) {
    const char test_data[] = "Hello, World!";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), sizeof(test_data) - 1);

    std::istream stream(&buffer);
    std::string result;
    stream >> result;

    EXPECT_EQ(result, "Hello,");

    stream >> result;
    EXPECT_EQ(result, "World!");
}

TEST(shared_stream_buffer, xsgetn) {
    const char test_data[] = "ABCDEFGHIJ";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), sizeof(test_data) - 1);

    char result[5] = {0};
    std::istream stream(&buffer);
    stream.read(result, 5);

    EXPECT_EQ(std::string(result, 5), "ABCDE");
    EXPECT_EQ(stream.gcount(), 5);

    stream.read(result, 5);
    EXPECT_EQ(std::string(result, 5), "FGHIJ");
    EXPECT_EQ(stream.gcount(), 5);
}

TEST(shared_stream_buffer, xsgetn_overflow) {
    const char test_data[] = "ABC";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 3);

    char result[10] = {0};
    std::istream stream(&buffer);
    stream.read(result, 10);

    EXPECT_EQ(stream.gcount(), 3);
    EXPECT_EQ(std::string(result, 3), "ABC");
}

TEST(shared_stream_buffer, underflow_and_uflow) {
    const char test_data[] = "AB";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 2);

    std::istream stream(&buffer);

    // Read using get() which uses uflow
    EXPECT_EQ(stream.get(), 'A');
    EXPECT_EQ(stream.get(), 'B');
    EXPECT_EQ(stream.get(), std::char_traits<char>::eof());
}

TEST(shared_stream_buffer, showmanyc) {
    const char test_data[] = "ABCDEFGH";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 8);

    std::istream stream(&buffer);
    EXPECT_EQ(stream.rdbuf()->in_avail(), 8);

    stream.get();
    EXPECT_EQ(stream.rdbuf()->in_avail(), 7);

    char tmp[5];
    stream.read(tmp, 5);
    EXPECT_EQ(stream.rdbuf()->in_avail(), 2);
}

TEST(shared_stream_buffer, seekoff_beg) {
    const char test_data[] = "0123456789";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 10);

    std::istream stream(&buffer);

    stream.seekg(5, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 5);
    EXPECT_EQ(stream.get(), '5');

    stream.seekg(0, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 0);
    EXPECT_EQ(stream.get(), '0');

    stream.seekg(9, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 9);
    EXPECT_EQ(stream.get(), '9');
}

TEST(shared_stream_buffer, seekoff_cur) {
    const char test_data[] = "0123456789";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 10);

    std::istream stream(&buffer);

    EXPECT_EQ(stream.get(), '0');

    stream.seekg(3, std::ios_base::cur);
    EXPECT_EQ(stream.tellg(), 4);
    EXPECT_EQ(stream.get(), '4');

    stream.seekg(-2, std::ios_base::cur);
    EXPECT_EQ(stream.tellg(), 3);
    EXPECT_EQ(stream.get(), '3');
}

TEST(shared_stream_buffer, seekoff_end) {
    const char test_data[] = "0123456789";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 10);

    std::istream stream(&buffer);

    stream.seekg(-1, std::ios_base::end);
    EXPECT_EQ(stream.tellg(), 9);
    EXPECT_EQ(stream.get(), '9');

    stream.seekg(-3, std::ios_base::end);
    EXPECT_EQ(stream.tellg(), 7);
    EXPECT_EQ(stream.get(), '7');

    stream.seekg(0, std::ios_base::end);
    EXPECT_EQ(stream.tellg(), 10);
    EXPECT_EQ(stream.get(), std::char_traits<char>::eof());
}

TEST(shared_stream_buffer, seekoff_invalid) {
    const char test_data[] = "0123456789";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 10);

    std::istream stream(&buffer);

    // Try to seek beyond buffer
    stream.seekg(20, std::ios_base::beg);
    EXPECT_TRUE(stream.fail());

    stream.clear();

    // Try to seek before beginning
    stream.seekg(-5, std::ios_base::beg);
    EXPECT_TRUE(stream.fail());
}

TEST(shared_stream_buffer, seekpos) {
    const char test_data[] = "0123456789";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 10);

    std::istream stream(&buffer);

    stream.seekg(7);
    EXPECT_EQ(stream.tellg(), 7);
    EXPECT_EQ(stream.get(), '7');

    stream.seekg(2);
    EXPECT_EQ(stream.tellg(), 2);
    EXPECT_EQ(stream.get(), '2');
}

TEST(shared_stream_buffer, large_size_no_memory) {
    // Test that SharedStreamBuffer can handle size > 2^32/2 (limitation of default streambuf implementations on
    // Windows)

    char dummy_data[10] = "test";
    constexpr size_t large_size = static_cast<size_t>(3000000000ULL);  // ~2.8GB, > 2^32/2

    // Create buffer claiming to be very large
    SharedStreamBuffer buffer(dummy_data, large_size);

    std::istream stream(&buffer);

    EXPECT_EQ(stream.rdbuf()->in_avail(), large_size);

    stream.seekg(1000000000, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 1000000000);

    stream.seekg(2000000000, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 2000000000);

    stream.seekg(-1000000000, std::ios_base::end);
    EXPECT_EQ(stream.tellg(), large_size - 1000000000);

    stream.seekg(0, std::ios_base::end);
    EXPECT_EQ(stream.tellg(), large_size);

    stream.seekg(0, std::ios_base::beg);
    EXPECT_EQ(stream.rdbuf()->in_avail(), large_size);
}

TEST(shared_stream_buffer, read_all_data) {
    const char test_data[] = "The quick brown fox jumps over the lazy dog";
    size_t data_size = sizeof(test_data) - 1;
    SharedStreamBuffer buffer(const_cast<char*>(test_data), data_size);

    std::istream stream(&buffer);
    std::string result;
    std::getline(stream, result);

    EXPECT_EQ(result, test_data);
}

TEST(shared_stream_buffer, sequential_operations) {
    const char test_data[] = "ABCDEFGHIJKLMNOP";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 16);

    std::istream stream(&buffer);

    char buf[5] = {0};
    stream.read(buf, 4);
    EXPECT_EQ(std::string(buf, 4), "ABCD");

    stream.seekg(8, std::ios_base::beg);
    EXPECT_EQ(stream.get(), 'I');

    stream.seekg(-5, std::ios_base::cur);
    EXPECT_EQ(stream.get(), 'E');

    stream.seekg(-2, std::ios_base::end);
    EXPECT_EQ(stream.get(), 'O');
}

TEST(shared_stream_buffer, negative_offset_from_current) {
    const char test_data[] = "0123456789ABCDEF";
    SharedStreamBuffer buffer(const_cast<char*>(test_data), 16);

    std::istream stream(&buffer);

    stream.seekg(10, std::ios_base::beg);
    EXPECT_EQ(stream.tellg(), 10);
    EXPECT_EQ(stream.get(), 'A');

    stream.seekg(-5, std::ios_base::cur);
    EXPECT_EQ(stream.tellg(), 6);
    EXPECT_EQ(stream.get(), '6');

    stream.seekg(-7, std::ios_base::cur);
    EXPECT_EQ(stream.tellg(), 0);
    EXPECT_EQ(stream.get(), '0');

    // Try to seek backward beyond beginning (should fail)
    stream.seekg(5, std::ios_base::beg);
    stream.clear();
    stream.seekg(-10, std::ios_base::cur);
    EXPECT_TRUE(stream.fail());
}
