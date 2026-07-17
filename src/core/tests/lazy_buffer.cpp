// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <string_view>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

using namespace testing;

namespace ov::test {
class LazyBufferTest : public Test {
protected:
    std::filesystem::path m_file_path;
    std::vector<char> m_test_data;

    void SetUp() override {
        m_file_path = utils::generateTestFilePrefix();
    }

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }

    void write_test_data(size_t size) {
        m_test_data.resize(size);
        std::iota(m_test_data.begin(), m_test_data.end(), 0);
        std::ofstream os(m_file_path, std::ios::binary);
        os.write(m_test_data.data(), m_test_data.size());
    }

    void overwrite_test_data(size_t offset, const std::vector<char>& data) {
        ASSERT_LE(offset + data.size(), m_test_data.size());
        std::copy(data.begin(), data.end(), m_test_data.begin() + offset);

        std::fstream fs(m_file_path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(fs.is_open());
        fs.seekp(offset);
        fs.write(data.data(), data.size());
        ASSERT_TRUE(fs.good());
    }
};

TEST_F(LazyBufferTest, incorrect_file) {
    OV_EXPECT_THROW(std::ignore = std::make_unique<LazyBuffer>(std::filesystem::path{"no_file"}, 1, 2),
                    AssertFailure,
                    HasSubstr("Failed to get file size"));

    write_test_data(4);

    const auto test_params =
        std::vector<std::tuple<size_t, size_t>>{{0, 5}, {1, 4}, {4, 2}, {0, std::numeric_limits<size_t>::max()}};
    for (const auto& [offset, size] : test_params) {
        OV_EXPECT_THROW(std::ignore = std::make_unique<LazyBuffer>(m_file_path, offset, size),
                        AssertFailure,
                        HasSubstr("exceeds file size"));
    }
}

TEST_F(LazyBufferTest, read_file) {
    write_test_data(457);
    const auto test_params = std::vector<std::tuple<size_t, size_t>>{{0, 10},
                                                                     {5, 20},
                                                                     {50, 100},
                                                                     {0, m_test_data.size()},
                                                                     {14, 15},
                                                                     {128, 256}};
    for (const auto& [offset, size] : test_params) {
        auto lazy_b = LazyBuffer{m_file_path, offset, size};
        AlignedBuffer& buffer = lazy_b;
        char* data_ptr = nullptr;
        ASSERT_NO_THROW((data_ptr = buffer.get_ptr<char>()));
        ASSERT_NE(data_ptr, nullptr);
        ASSERT_EQ(buffer.size(), size);
        EXPECT_THAT(std::string_view(data_ptr, size), ElementsAreArray(m_test_data.data() + offset, size));
    }
}

TEST_F(LazyBufferTest, load_on_first_get_ptr) {
    write_test_data(128);

    constexpr size_t offset = 37;
    constexpr size_t size = 9;
    const std::vector<char> first_rewrite{'L', 'A', 'Z', 'Y', ' ', 'D', 'A', 'T', 'A'};
    const std::vector<char> second_rewrite{'O', 'V', 'E', 'R', 'W', 'R', 'I', 'T', 'E'};

    std::unique_ptr<AlignedBuffer> buffer = std::make_unique<LazyBuffer>(m_file_path, offset, size);

    // If constructor eagerly reads file data, get_ptr() would return original bytes instead of first_rewrite.
    overwrite_test_data(offset, first_rewrite);

    char* first_ptr = nullptr;
    ASSERT_NO_THROW((first_ptr = buffer->get_ptr<char>()));
    ASSERT_NE(first_ptr, nullptr);
    ASSERT_TRUE(std::equal(first_ptr, first_ptr + size, first_rewrite.begin()));

    // Once loaded, subsequent get_ptr() calls should return cached memory and ignore later file overwrites.
    overwrite_test_data(offset, second_rewrite);

    char* second_ptr = nullptr;
    ASSERT_NO_THROW((second_ptr = buffer->get_ptr<char>()));
    ASSERT_EQ(second_ptr, first_ptr);
    EXPECT_THAT(first_rewrite, ElementsAreArray(second_ptr, size));
}

TEST_F(LazyBufferTest, evict_and_reload) {
    write_test_data(128);

    constexpr size_t offset = 31;
    constexpr size_t size = 9;
    const std::vector<char> first_rewrite{'L', 'A', 'Z', 'Y', ' ', 'D', 'A', 'T', 'A'};
    const std::vector<char> second_rewrite{'O', 'V', 'E', 'R', 'W', 'R', 'I', 'T', 'E'};

    const auto buffer = std::make_unique<LazyBuffer>(m_file_path, offset, size);

    overwrite_test_data(offset, first_rewrite);
    char* first_ptr = nullptr;
    ASSERT_NO_THROW((first_ptr = buffer->get_ptr<char>()));
    ASSERT_NE(first_ptr, nullptr);
    ASSERT_THAT(first_rewrite, ElementsAreArray(first_ptr, size));

    buffer->hint_evict();
    ASSERT_NO_THROW((first_ptr = buffer->get_ptr<char>()));
    ASSERT_NE(first_ptr, nullptr);
    ASSERT_THAT(first_rewrite, ElementsAreArray(first_ptr, size));

    buffer->hint_evict();
    overwrite_test_data(offset, second_rewrite);
    char* second_ptr = nullptr;
    ASSERT_NO_THROW((second_ptr = buffer->get_ptr<char>()));
    ASSERT_EQ(second_ptr, first_ptr);
    EXPECT_THAT(second_rewrite, ElementsAreArray(second_ptr, size));
}

TEST_F(LazyBufferTest, move_constructor_unloaded) {
    write_test_data(128);
    constexpr size_t offset = 10;
    constexpr size_t size = 50;

    LazyBuffer source{m_file_path, offset, size};
    const auto src_buf_ptr = source.get_ptr<char>();

    LazyBuffer dest{std::move(source)};

    ASSERT_EQ(dest.size(), size);
    const auto dest_buf_ptr = dest.get_ptr<char>();
    ASSERT_EQ(dest_buf_ptr, src_buf_ptr);
    EXPECT_THAT(std::string_view(dest_buf_ptr, size), ElementsAreArray(m_test_data.data() + offset, size));
}

TEST_F(LazyBufferTest, move_constructor_loaded) {
    write_test_data(128);
    constexpr size_t offset = 5;
    constexpr size_t size = 30;

    LazyBuffer source{m_file_path, offset, size};
    const auto src_buf_ptr = source.get_ptr<char>();
    const std::vector<char> expected(src_buf_ptr, src_buf_ptr + size);

    // Overwrite file after prefetch; moved object should serve cached data, not re-read
    overwrite_test_data(offset, std::vector<char>(size, 0xFF));

    LazyBuffer dest{std::move(source)};

    ASSERT_EQ(dest.size(), size);
    const auto dest_buf_ptr = dest.get_ptr<char>();
    ASSERT_EQ(dest_buf_ptr, src_buf_ptr);
    EXPECT_THAT(std::string_view(dest_buf_ptr, size), ElementsAreArray(expected));
}

TEST_F(LazyBufferTest, move_assignment_unloaded) {
    write_test_data(128);
    constexpr size_t offset = 20;
    constexpr size_t size = 40;

    LazyBuffer source{m_file_path, offset, size};
    const auto src_buf_ptr = source.get_ptr<char>();

    LazyBuffer dest{m_file_path, 0, 5};
    dest = std::move(source);

    ASSERT_EQ(dest.size(), size);
    const auto dest_buf_ptr = dest.get_ptr<char>();
    ASSERT_EQ(dest_buf_ptr, src_buf_ptr);
    EXPECT_THAT(std::string_view(dest_buf_ptr, size), ElementsAreArray(m_test_data.data() + offset, size));
}

TEST_F(LazyBufferTest, move_assignment_loaded) {
    write_test_data(128);
    constexpr size_t offset = 15;
    constexpr size_t size = 20;

    LazyBuffer source{m_file_path, offset, size};

    const auto src_buf_ptr = source.get_ptr<char>();
    const std::vector<char> expected(src_buf_ptr, src_buf_ptr + size);

    // Overwrite file after prefetch; dest should keep cached data, not re-read
    overwrite_test_data(offset, std::vector<char>(size, 0xFF));

    LazyBuffer dest{m_file_path, 0, 5};
    dest = std::move(source);

    ASSERT_EQ(dest.size(), size);
    const auto dest_buf_ptr = dest.get_ptr<char>();
    ASSERT_EQ(dest_buf_ptr, src_buf_ptr);
    EXPECT_THAT(std::string_view(dest_buf_ptr, size), ElementsAreArray(expected));
}

TEST_F(LazyBufferTest, move_assignment_evict_and_reload) {
    write_test_data(128);
    constexpr size_t offset = 10;
    constexpr size_t size = 30;

    LazyBuffer source{m_file_path, offset, size};
    LazyBuffer dest{m_file_path, 0, 5};
    dest = std::move(source);

    ASSERT_NO_THROW(dest.hint_prefetch());
    dest.hint_evict();

    const auto data_ptr = dest.get_ptr<char>();
    EXPECT_THAT(std::string_view(data_ptr, size), ElementsAreArray(m_test_data.data() + offset, size));
}

TEST_F(LazyBufferTest, move_assignment_self_no_op) {
    write_test_data(64);

    LazyBuffer buf{m_file_path, 0, 64};
    const auto src_buf_ptr = buf.get_ptr<char>();
    const auto buf_size = buf.size();

    auto& buf_ref = buf;
    buf = std::move(buf_ref);

    EXPECT_EQ(buf.get_ptr<char>(), src_buf_ptr);
    EXPECT_EQ(buf.size(), buf_size);
}
}  // namespace ov::test
