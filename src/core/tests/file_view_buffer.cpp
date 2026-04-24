// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/file_view_buffer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

namespace ov::test {
class FileViewBufferTest : public ::testing::Test {
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
};

TEST_F(FileViewBufferTest, incorrect_file) {
    OV_EXPECT_THROW(std::ignore = std::make_unique<FileViewBuffer>(std::filesystem::path{"no_file"}, 1, 2),
                    AssertFailure,
                    ::testing::HasSubstr("File does not exist"));

    write_test_data(4);

    const auto test_params = std::vector<std::tuple<size_t, size_t>>{{0, 5}, {1, 4}, {4, 2}};
    for (const auto& [offset, size] : test_params) {
        OV_EXPECT_THROW(std::ignore = std::make_unique<FileViewBuffer>(m_file_path, offset, size),
                        AssertFailure,
                        ::testing::HasSubstr("File size is smaller than the requested view"));
    }
}

TEST_F(FileViewBufferTest, read_file) {
    write_test_data(457);
    const auto test_params = std::vector<std::tuple<size_t, size_t, size_t>>{{0, 10, 64},
                                                                             {5, 20, 64},
                                                                             {50, 100, 64},
                                                                             {0, m_test_data.size(), 64},
                                                                             {14, 15, 29}};
    for (const auto& [offset, size, alignment] : test_params) {
        std::unique_ptr<AlignedBuffer> buffer = std::make_unique<FileViewBuffer>(m_file_path, offset, size, alignment);
        char* data_ptr = nullptr;
        ASSERT_NO_THROW((data_ptr = buffer->get_ptr<char>()));
        ASSERT_NE(data_ptr, nullptr);
        ASSERT_EQ(buffer->size(), size);
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data_ptr) % alignment, 0);
        EXPECT_TRUE(std::equal(data_ptr, data_ptr + size, m_test_data.begin() + offset));
    }
}
}  // namespace ov::test
