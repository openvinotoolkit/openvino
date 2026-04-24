// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

namespace ov::test {
class LazyBufferTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;
    std::vector<char> m_test_data;

    void SetUp() override {
        m_file_path = utils::generateTestFilePrefix();
    }

    void TearDown() override {
        std::filesystem::remove(m_file_path);
    }

public:
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
                    ::testing::HasSubstr("File does not exist"));

    write_test_data(4);

    const auto test_params = std::vector<std::tuple<size_t, size_t>>{{0, 5}, {1, 4}, {4, 2}};
    for (const auto& [offset, size] : test_params) {
        OV_EXPECT_THROW(std::ignore = std::make_unique<LazyBuffer>(m_file_path, offset, size),
                        AssertFailure,
                        ::testing::HasSubstr("File size is smaller than the requested view"));
    }
}

TEST_F(LazyBufferTest, read_file) {
    write_test_data(457);
    const auto test_params = std::vector<std::tuple<size_t, size_t, size_t>>{{0, 10, 64},
                                                                             {5, 20, 64},
                                                                             {50, 100, 64},
                                                                             {0, m_test_data.size(), 64},
                                                                             {14, 15, 29}};
    for (const auto& [offset, size, alignment] : test_params) {
        std::unique_ptr<AlignedBuffer> buffer = std::make_unique<LazyBuffer>(m_file_path, offset, size, alignment);
        char* data_ptr = nullptr;
        ASSERT_NO_THROW((data_ptr = buffer->get_ptr<char>()));
        ASSERT_NE(data_ptr, nullptr);
        ASSERT_EQ(buffer->size(), size);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data_ptr) % alignment, 0);
        EXPECT_TRUE(std::equal(data_ptr, data_ptr + size, m_test_data.begin() + offset));
    }
}

TEST_F(LazyBufferTest, load_happens_on_first_get_ptr) {
    write_test_data(128);

    constexpr size_t offset = 32;
    constexpr size_t size = 8;
    const std::vector<char> first_rewrite{'L', 'A', 'Z', 'Y', 'D', 'A', 'T', 'A'};
    const std::vector<char> second_rewrite{'E', 'A', 'G', 'E', 'R', 'D', 'A', 'T'};

    std::unique_ptr<AlignedBuffer> buffer = std::make_unique<LazyBuffer>(m_file_path, offset, size, 64);

    // If constructor eagerly reads file data, get_ptr() would return original bytes instead of first_rewrite.
    overwrite_test_data(offset, first_rewrite);

    char* first_ptr = nullptr;
    ASSERT_NO_THROW((first_ptr = buffer->get_ptr<char>()));
    ASSERT_NE(first_ptr, nullptr);
    ASSERT_TRUE(std::equal(first_ptr, first_ptr + size, first_rewrite.begin()));

    // Once loaded, subsequent get_ptr() calls should return cached memory and ignore later file rewrites.
    overwrite_test_data(offset, second_rewrite);

    char* second_ptr = nullptr;
    ASSERT_NO_THROW((second_ptr = buffer->get_ptr<char>()));
    ASSERT_EQ(second_ptr, first_ptr);
    EXPECT_TRUE(std::equal(second_ptr, second_ptr + size, first_rewrite.begin()));
}
}  // namespace ov::test
