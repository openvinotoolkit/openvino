// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"

#include <sstream>

#include "gtest/gtest.h"
#include "common_test_utils/common_utils.hpp"
#include "openvino/util/mmap_object.hpp"

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

// ==================== SharedBuffer Tests ====================

class SharedBufferTest : public ::testing::Test {
protected:
    static constexpr size_t test_data_size = 100;
    char test_data[test_data_size] = "Test data for SharedBuffer tests";

    std::string m_file_path;

    void SetUp() override {
        m_file_path = ov::test::utils::generateTestFilePrefix();
    }

    void TearDown() override {
        std::remove(m_file_path.c_str());
    }

    void create_file() {
        std::ofstream os(m_file_path, std::ios::binary);
        os.write(test_data, test_data_size);
        os.close();
    }
};

// Test basic SharedBuffer with shared_ptr<void> - no descriptor
TEST_F(SharedBufferTest, basic_shared_ptr_void_no_descriptor) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);

    EXPECT_EQ(buffer->size(), test_data_size);
    EXPECT_EQ(buffer->get_ptr(), test_data);

    // No descriptor provided, so get_descriptor() should return nullptr
    EXPECT_EQ(buffer->get_descriptor(), nullptr);
}

// Test SharedBuffer<MappedMemory> auto-creates descriptor via MMapDescriptor
TEST_F(SharedBufferTest, mapped_memory_creates_descriptor) {
    create_file();
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data(), mapped_memory->size(), mapped_memory);

    auto desc = buffer->get_descriptor();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->get_id(), mapped_memory->get_id());
    EXPECT_NE(desc->get_id(), 0u);
    EXPECT_EQ(desc->get_offset(), 0u);

    // get_source_buffer should return a buffer backed by the MappedMemory
    auto source = desc->get_source_buffer();
    ASSERT_NE(source, nullptr);
    EXPECT_EQ(source->get_ptr(), mapped_memory->data());
    EXPECT_EQ(source->size(), mapped_memory->size());
}

// Test get_descriptor returns nullptr for regular AlignedBuffer
TEST_F(SharedBufferTest, regular_aligned_buffer_no_descriptor) {
    ov::AlignedBuffer buffer(100);
    EXPECT_EQ(buffer.get_descriptor(), nullptr);
}

// Test SharedBuffer lifetime - shared object should be kept alive
TEST_F(SharedBufferTest, shared_object_lifetime) {
    std::weak_ptr<int> weak_obj;
    {
        auto shared_obj = std::make_shared<int>(42);
        weak_obj = shared_obj;

        auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
            test_data, test_data_size, shared_obj);

        // shared_obj goes out of scope, but buffer should keep it alive
        shared_obj.reset();
        EXPECT_FALSE(weak_obj.expired());
    }
    // buffer goes out of scope, now the object should be destroyed
    EXPECT_TRUE(weak_obj.expired());
}

// Test AlignedBuffer interface
TEST_F(SharedBufferTest, aligned_buffer_interface) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);

    // Access through AlignedBuffer interface
    ov::AlignedBuffer* aligned = buffer.get();
    EXPECT_EQ(aligned->size(), test_data_size);
    EXPECT_EQ(aligned->get_ptr(), test_data);
}

// ==================== MappedMemory get_id Tests ====================

TEST(MappedMemory, get_id_unique_per_file) {
    // Create two temporary files
    std::string file1 = ov::test::utils::generateTestFilePrefix() + "_file1";
    std::string file2 = ov::test::utils::generateTestFilePrefix() + "_file2";

    const char test_data[] = "Test data for MappedMemory";

    // Write same data to both files
    {
        std::ofstream os1(file1, std::ios::binary);
        os1.write(test_data, sizeof(test_data));
        os1.close();

        std::ofstream os2(file2, std::ios::binary);
        os2.write(test_data, sizeof(test_data));
        os2.close();
    }

    // Load both files
    auto mapped1 = ov::load_mmap_object(file1);
    auto mapped2 = ov::load_mmap_object(file2);

    ASSERT_NE(mapped1, nullptr);
    ASSERT_NE(mapped2, nullptr);

    // IDs should be different even though content is the same (ID is hash of file path)
    EXPECT_NE(mapped1->get_id(), mapped2->get_id());

    // Clean up
    std::remove(file1.c_str());
    std::remove(file2.c_str());
}

TEST(MappedMemory, get_id_same_for_same_file) {
    std::string file_path = ov::test::utils::generateTestFilePrefix() + "_same_file";
    const char test_data[] = "Test data for same file";

    // Create file
    {
        std::ofstream os(file_path, std::ios::binary);
        os.write(test_data, sizeof(test_data));
        os.close();
    }

    // Load the same file twice
    auto mapped1 = ov::load_mmap_object(file_path);
    auto mapped2 = ov::load_mmap_object(file_path);

    ASSERT_NE(mapped1, nullptr);
    ASSERT_NE(mapped2, nullptr);

    // IDs should be the same for the same file path
    EXPECT_EQ(mapped1->get_id(), mapped2->get_id());

    // Clean up
    std::remove(file_path.c_str());
}

TEST(MappedMemory, get_id_non_zero) {
    std::string file_path = ov::test::utils::generateTestFilePrefix() + "_non_zero";
    const char test_data[] = "Test data";

    // Create file
    {
        std::ofstream os(file_path, std::ios::binary);
        os.write(test_data, sizeof(test_data));
        os.close();
    }

    auto mapped = ov::load_mmap_object(file_path);
    ASSERT_NE(mapped, nullptr);

    // ID should be non-zero since it's computed as hash of file path
    EXPECT_NE(mapped->get_id(), 0u);
    EXPECT_NE(mapped->get_id(), std::numeric_limits<uint64_t>::max());

    // Clean up
    std::remove(file_path.c_str());
}
