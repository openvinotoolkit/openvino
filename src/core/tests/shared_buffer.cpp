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

// ==================== SharedBuffer Tests ====================

class SharedBufferTest : public ::testing::Test {
protected:
    static constexpr size_t test_data_size = 100;
    char test_data[test_data_size] = "Test data for SharedBuffer tests";
};

// Test basic SharedBuffer with shared_ptr<void>
TEST_F(SharedBufferTest, basic_shared_ptr_void) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);

    EXPECT_EQ(buffer->size(), test_data_size);
    EXPECT_EQ(buffer->get_ptr(), test_data);
    EXPECT_EQ(buffer->is_mapped(), false);
    EXPECT_EQ(buffer->get_tag(), "");
    EXPECT_EQ(buffer->get_id(), 0);
    EXPECT_EQ(buffer->get_offset(), 0);
}

// Test SharedBuffer with tag
TEST_F(SharedBufferTest, with_tag) {
    auto shared_obj = std::make_shared<int>(42);
    const std::string tag = "test_tag_123";
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj, tag);

    EXPECT_EQ(buffer->get_tag(), tag);
    EXPECT_NE(buffer->get_id(), 0);
    EXPECT_EQ(buffer->get_id(), std::hash<std::string>{}(tag));
}

// Test SharedBuffer with empty tag
TEST_F(SharedBufferTest, with_empty_tag) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj, "");

    EXPECT_EQ(buffer->get_tag(), "");
    EXPECT_EQ(buffer->get_id(), 0);
}

// Test is_mapped with MappedMemory
TEST_F(SharedBufferTest, is_mapped_with_mapped_memory) {
    auto mapped_memory = ov::load_mmap_object("/home/oleg/workspace/openvino/src/core/tests/shared_buffer.cpp");
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data(), mapped_memory->size(), mapped_memory);

    EXPECT_TRUE(buffer->is_mapped());
}

// Test is_mapped returns false for non-mapped buffer
TEST_F(SharedBufferTest, is_mapped_false_for_non_mapped) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);

    EXPECT_FALSE(buffer->is_mapped());
}

// Test nested SharedBuffer - is_mapped propagates through chain
TEST_F(SharedBufferTest, is_mapped_propagates_through_nested_buffers) {
    // Create a mapped memory buffer
    auto mapped_memory = ov::load_mmap_object("/home/oleg/workspace/openvino/src/core/tests/shared_buffer.cpp");
    auto inner_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data(), mapped_memory->size(), mapped_memory);
    EXPECT_TRUE(inner_buffer->is_mapped());

    // Wrap it in another SharedBuffer with AlignedBuffer type
    auto outer_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        mapped_memory->data(), mapped_memory->size(), inner_buffer);
    EXPECT_TRUE(outer_buffer->is_mapped());

    // Wrap again
    auto outer_buffer2 = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        mapped_memory->data(), mapped_memory->size(), outer_buffer);
    EXPECT_TRUE(outer_buffer2->is_mapped());
}

// Test nested SharedBuffer - is_mapped returns false when inner is not mapped
TEST_F(SharedBufferTest, is_mapped_false_for_nested_non_mapped) {
    auto shared_obj = std::make_shared<int>(42);
    auto inner_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);
    EXPECT_FALSE(inner_buffer->is_mapped());

    auto outer_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        test_data, test_data_size, inner_buffer);
    EXPECT_FALSE(outer_buffer->is_mapped());
}

// Test get_offset with AlignedBuffer (which has get_ptr method)
TEST_F(SharedBufferTest, get_offset_with_aligned_buffer) {
    auto mapped_memory = ov::load_mmap_object("/home/oleg/workspace/openvino/src/core/tests/shared_buffer.cpp");
    auto inner_buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data(), mapped_memory->size(), mapped_memory);

    // Buffer at the beginning - offset should be 0
    auto buffer_start = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        mapped_memory->data(), mapped_memory->size(), inner_buffer);
    EXPECT_EQ(buffer_start->get_offset(), 0);

    // Buffer with offset
    constexpr size_t offset = 100;
    if (mapped_memory->size() > offset) {
        auto buffer_offset = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
            mapped_memory->data() + offset, mapped_memory->size() - offset, inner_buffer);
        EXPECT_EQ(buffer_offset->get_offset(), offset);
    }
}

// Test get_offset returns 0 for MappedMemory (no get_ptr method)
TEST_F(SharedBufferTest, get_offset_zero_for_mapped_memory) {
    auto mapped_memory = ov::load_mmap_object("/home/oleg/workspace/openvino/src/core/tests/shared_buffer.cpp");
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        mapped_memory->data(), mapped_memory->size(), mapped_memory);

    // MappedMemory doesn't have get_ptr<T>() method, so get_offset returns 0
    EXPECT_EQ(buffer->get_offset(), 0);
}

// Test get_offset returns 0 for non-offsetable types
TEST_F(SharedBufferTest, get_offset_zero_for_void_ptr) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj);

    EXPECT_EQ(buffer->get_offset(), 0);
}

// Test different tags produce different IDs
TEST_F(SharedBufferTest, different_tags_different_ids) {
    auto shared_obj = std::make_shared<int>(42);

    auto buffer1 = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj, "tag1");
    auto buffer2 = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj, "tag2");

    EXPECT_NE(buffer1->get_id(), buffer2->get_id());
    EXPECT_NE(buffer1->get_tag(), buffer2->get_tag());
}

// Test same tags produce same IDs
TEST_F(SharedBufferTest, same_tags_same_ids) {
    auto shared_obj1 = std::make_shared<int>(42);
    auto shared_obj2 = std::make_shared<int>(43);
    const std::string tag = "same_tag";

    auto buffer1 = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj1, tag);
    auto buffer2 = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj2, tag);

    EXPECT_EQ(buffer1->get_id(), buffer2->get_id());
    EXPECT_EQ(buffer1->get_tag(), buffer2->get_tag());
}

// Test SharedBuffer with non-polymorphic shared_ptr type
TEST_F(SharedBufferTest, shared_ptr_non_polymorphic_type) {
    struct NonPolymorphic {
        int value;
    };

    auto shared_obj = std::make_shared<NonPolymorphic>();
    shared_obj->value = 123;

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<NonPolymorphic>>>(
        test_data, test_data_size, shared_obj);

    // Should compile and work, is_mapped should return false
    EXPECT_FALSE(buffer->is_mapped());
    EXPECT_EQ(buffer->size(), test_data_size);
}

// Test SharedBuffer with polymorphic shared_ptr type that is not ITagBuffer
TEST_F(SharedBufferTest, shared_ptr_polymorphic_non_itag_buffer) {
    struct PolymorphicNonTag {
        virtual ~PolymorphicNonTag() = default;
        int value;
    };

    auto shared_obj = std::make_shared<PolymorphicNonTag>();
    shared_obj->value = 456;

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<PolymorphicNonTag>>>(
        test_data, test_data_size, shared_obj);

    // Should compile and work, is_mapped should return false (dynamic_cast will return nullptr)
    EXPECT_FALSE(buffer->is_mapped());
    EXPECT_EQ(buffer->size(), test_data_size);
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

// Test ITagBuffer interface through base pointer
TEST_F(SharedBufferTest, itag_buffer_interface) {
    auto shared_obj = std::make_shared<int>(42);
    const std::string tag = "interface_test";

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(
        test_data, test_data_size, shared_obj, tag);

    // Access through ITagBuffer interface
    ov::ITagBuffer* itag = buffer.get();
    EXPECT_EQ(itag->get_tag(), tag);
    EXPECT_EQ(itag->get_id(), std::hash<std::string>{}(tag));
    EXPECT_FALSE(itag->is_mapped());
    EXPECT_EQ(itag->get_offset(), 0);
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
