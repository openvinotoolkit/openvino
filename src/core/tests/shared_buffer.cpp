// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/shared_buffer.hpp"

#include <gmock/gmock.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::test {
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

    std::filesystem::path m_file_path;

    void SetUp() override {
        m_file_path = ov::test::utils::generateTestFilePrefix();
        std::ofstream os(m_file_path, std::ios::binary);
        os.write(test_data, test_data_size);
    }

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }
};

// Test basic SharedBuffer with shared_ptr<void> - no descriptor
TEST_F(SharedBufferTest, basic_shared_ptr_void_no_descriptor) {
    auto shared_obj = std::make_shared<int>(42);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(test_data, test_data_size, shared_obj);

    EXPECT_EQ(buffer->size(), test_data_size);
    EXPECT_EQ(buffer->get_ptr(), test_data);

    // No descriptor provided, so get_descriptor() should return nullptr
    EXPECT_EQ(buffer->get_descriptor(), nullptr);
}

// Test SharedBuffer<MappedMemory> auto-creates descriptor via MMapDescriptor
TEST_F(SharedBufferTest, mapped_memory_creates_descriptor) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                        mapped_memory->size(),
                                                                                        mapped_memory);

    auto desc = buffer->get_descriptor();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->get_id(), mapped_memory->get_id());
    EXPECT_NE(desc->get_id(), 0u);
    EXPECT_EQ(desc->get_offset(), 0u);

    std::weak_ptr<ov::AlignedBuffer> weak_source;
    { weak_source = desc->get_source_buffer(); }
    auto source = weak_source.lock();
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

        auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(test_data, test_data_size, shared_obj);

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
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(test_data, test_data_size, shared_obj);

    // Access through AlignedBuffer interface
    ov::AlignedBuffer* aligned = buffer.get();
    EXPECT_EQ(aligned->size(), test_data_size);
    EXPECT_EQ(aligned->get_ptr(), test_data);
}

// ==================== SharedBuffer with explicit descriptor Tests ====================

TEST_F(SharedBufferTest, shared_ptr_void_with_explicit_descriptor) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    // Create a source buffer from mapped memory
    auto source = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(mapped_memory->data(),
                                                                            mapped_memory->size(),
                                                                            mapped_memory);

    // Create a descriptor manually
    auto descriptor = ov::create_base_descriptor(42, 10, source);

    // Create SharedBuffer with explicit descriptor
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(mapped_memory->data(),
                                                                            mapped_memory->size(),
                                                                            std::shared_ptr<void>{},
                                                                            descriptor);

    auto desc = buffer->get_descriptor();
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->get_id(), 42u);

    // The data in descriptor should match the pointer and size of the buffer
    EXPECT_THROW(std::ignore = std::make_shared<ov::SharedBuffer<std::shared_ptr<void>>>(test_data,
                                                                                         test_data_size,
                                                                                         std::shared_ptr<void>{},
                                                                                         descriptor),
                 ov::Exception);
}

// ==================== SharedBuffer<shared_ptr<AlignedBuffer>> subbuffer Tests ====================
TEST_F(SharedBufferTest, aligned_buffer_derived_auto_inherits_descriptor) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    // Create a parent SharedBuffer from MappedMemory (has descriptor)
    auto parent = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                        mapped_memory->size(),
                                                                                        mapped_memory);

    auto parent_desc = parent->get_descriptor();
    ASSERT_NE(parent_desc, nullptr);

    // Create SharedBuffer<shared_ptr<AlignedBuffer>> using 2-arg data/size constructor
    // which should auto-inherit descriptor from the shared object
    auto child = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        static_cast<char*>(parent->get_ptr()) + 5,
        20,
        std::static_pointer_cast<ov::AlignedBuffer>(parent));

    auto child_desc = child->get_descriptor();
    ASSERT_NE(child_desc, nullptr);
    EXPECT_EQ(child_desc->get_id(), parent_desc->get_id());
    EXPECT_EQ(child_desc->get_offset(), 5u);
}

TEST_F(SharedBufferTest, mmap_with_offset) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    const auto offset = 10u;
    auto parent = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + offset,
                                                                                        mapped_memory->size() - offset,
                                                                                        mapped_memory);

    auto parent_desc = parent->get_descriptor();
    ASSERT_NE(parent_desc, nullptr);
    EXPECT_EQ(parent_desc->get_offset(), offset);
}

TEST_F(SharedBufferTest, mmap_source_buffer) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    const auto offset = 10u;
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + offset,
                                                                                        mapped_memory->size() - offset,
                                                                                        mapped_memory);
    auto buffer_desc = buffer->get_descriptor();
    auto source_buffer = buffer_desc->get_source_buffer();
    ASSERT_NE(buffer_desc, nullptr);
    EXPECT_EQ(buffer_desc->get_id(), mapped_memory->get_id());
    EXPECT_EQ(buffer_desc->get_offset(), offset);
    ASSERT_NE(source_buffer, nullptr);

    auto source_desc = source_buffer->get_descriptor();
    ASSERT_NE(source_desc, nullptr);
    EXPECT_EQ(source_desc->get_id(), mapped_memory->get_id());
    EXPECT_EQ(source_desc->get_offset(), 0u);
}

TEST_F(SharedBufferTest, mmap_nested_buffer) {
    auto mapped_memory = ov::load_mmap_object(m_file_path);

    const auto offset = 10u;
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data() + offset,
                                                                                        mapped_memory->size() - offset,
                                                                                        mapped_memory);
    auto nested_buffer =
        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(buffer->get_ptr<char>() + offset,
                                                                               buffer->size() - offset,
                                                                               buffer);

    auto nested_buffer_desc = nested_buffer->get_descriptor();
    auto source_buffer = nested_buffer_desc->get_source_buffer();
    ASSERT_NE(nested_buffer_desc, nullptr);
    EXPECT_EQ(nested_buffer_desc->get_id(), mapped_memory->get_id());
    EXPECT_EQ(nested_buffer_desc->get_offset(), offset * 2);
    ASSERT_NE(source_buffer, nullptr);

    auto source_desc = source_buffer->get_descriptor();
    ASSERT_NE(source_desc, nullptr);
    EXPECT_EQ(source_desc->get_id(), mapped_memory->get_id());
    EXPECT_EQ(source_desc->get_offset(), 0u);
}

TEST_F(SharedBufferTest, specialization_overload_resolution) {
    // Test various SharedBuffer constructor variants to ensure SFINAE and overload resolution work correctly
    {
        auto src_buffer = std::make_shared<ov::AlignedBuffer>(sizeof(float) * 4000);
        {
            auto sh_buff = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
                src_buffer->get_ptr<char>(),
                src_buffer->size(),
                src_buffer,
                ov::create_base_descriptor(10, 5, src_buffer));

            auto desc = sh_buff->get_descriptor();
            ASSERT_NE(desc, nullptr);
            EXPECT_EQ(desc->get_id(), 10u);
            EXPECT_EQ(desc->get_offset(),
                      0u);  // offset is calculated from source buffer pointer, not constructor argument
            EXPECT_EQ(desc->get_source_buffer(), src_buffer);
        }
        {
            auto sh_buff =
                std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(src_buffer->get_ptr<char>(),
                                                                                       src_buffer->size(),
                                                                                       src_buffer);

            EXPECT_EQ(sh_buff->get_descriptor(), nullptr);  // no descriptor to inherit
        }
    }
    {
        auto mapped_memory = ov::load_mmap_object(m_file_path);
        {
            auto sh_buff = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
                mapped_memory->data(),
                mapped_memory->size(),
                mapped_memory,
                ov::create_base_descriptor(10, 5, nullptr));

            auto desc = sh_buff->get_descriptor();
            ASSERT_NE(desc, nullptr);
            EXPECT_EQ(desc->get_id(), 10u);
            EXPECT_EQ(desc->get_offset(), 0u);              // no source buffer provided, so offset should be 0
            EXPECT_EQ(desc->get_source_buffer(), nullptr);  // no source buffer provided
        }
        {
            auto sh_buff = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mapped_memory->data(),
                                                                                                 mapped_memory->size(),
                                                                                                 mapped_memory);

            auto desc = sh_buff->get_descriptor();
            ASSERT_NE(desc, nullptr);
            EXPECT_EQ(desc->get_id(), mapped_memory->get_id());
            EXPECT_EQ(desc->get_offset(), 0u);
            EXPECT_NE(desc->get_source_buffer(), nullptr);
        }
    }
}

class MockMappedMemory : public ov::MappedMemory {
public:
    explicit MockMappedMemory(size_t size) : m_data(size, '\0'), m_id(1) {}

    char* data() noexcept override {
        return m_data.data();
    }
    size_t size() const noexcept override {
        return m_data.size();
    }
    uint64_t get_id() const noexcept override {
        return m_id;
    }

    MOCK_METHOD(void, hint_evict, (size_t offset, size_t size), (override));

private:
    std::vector<char> m_data;
    uint64_t m_id;
};

TEST_F(SharedBufferTest, mmap_shared_buffer_calls_hint_evict_with_own_region) {
    constexpr size_t mmap_size = 1024;
    constexpr size_t buf_offset = 128;
    constexpr size_t buf_size = 256;

    auto mock = std::make_shared<MockMappedMemory>(mmap_size);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mock->data() + buf_offset,
                                                                                        buf_size,
                                                                                        mock);

    EXPECT_CALL(*mock, hint_evict(buf_offset, buf_size)).Times(1);
    buffer->hint_evict();
}

TEST_F(SharedBufferTest, mmap_shared_buffer_full_mapping) {
    constexpr size_t mmap_size = 512;

    auto mock = std::make_shared<MockMappedMemory>(mmap_size);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mock->data(), mmap_size, mock);

    EXPECT_CALL(*mock, hint_evict(0u, mmap_size)).Times(1);
    buffer->hint_evict();
}

TEST_F(SharedBufferTest, aligned_shared_buffer_propagates_to_mmap) {
    constexpr size_t mmap_size = 2048;
    constexpr size_t parent_offset = 64;
    constexpr size_t child_offset = 32;  // relative to parent data ptr
    constexpr size_t child_size = 128;

    auto mock = std::make_shared<MockMappedMemory>(mmap_size);

    auto parent = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mock->data() + parent_offset,
                                                                                        mmap_size - parent_offset,
                                                                                        mock);

    auto child = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(
        parent->get_ptr<char>() + child_offset,
        child_size,
        std::static_pointer_cast<ov::AlignedBuffer>(parent));

    // child lives at parent_offset + child_offset inside the mmap
    EXPECT_CALL(*mock, hint_evict(parent_offset + child_offset, child_size)).Times(1);
    child->hint_evict();
}

TEST_F(SharedBufferTest, no_call_when_mmap_object_is_null) {
    constexpr size_t buf_size = 64;
    std::vector<char> storage(buf_size);

    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(
        storage.data(),
        buf_size,
        std::shared_ptr<ov::MappedMemory>{} /*null*/);
    EXPECT_NO_THROW(buffer->hint_evict());
}

TEST_F(SharedBufferTest, call_when_constant_node_destroyed) {
    constexpr size_t mmap_size = 1024;
    auto mock = std::make_shared<MockMappedMemory>(mmap_size);
    auto buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::MappedMemory>>>(mock->data(), mmap_size, mock);

    EXPECT_CALL(*mock, hint_evict(0u, mmap_size)).Times(1);
    {
        auto constant = op::v0::Constant(element::u8, Shape{mmap_size}, buffer);
        EXPECT_EQ(constant.get_data_ptr(), buffer->get_ptr());
    }
}

}  // namespace ov::test
