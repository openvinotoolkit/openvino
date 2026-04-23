// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/file_view_buffer.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"

namespace ov::test {
/* class FileViewBufferTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;
    std::vector<char> m_test_data;

    void SetUp() override {}

    void TearDown() override {}
};
 */
TEST(FileViewBufferTest, dummy_file) {
    const auto dummy_file = std::filesystem::path{"non_existent_file.bin"};
    std::unique_ptr<FileViewBuffer> view_buffer;
    EXPECT_NO_THROW(view_buffer = std::make_unique<FileViewBuffer>(dummy_file, 1, 2));
    EXPECT_NO_THROW(view_buffer->release());
    EXPECT_THROW(view_buffer->load(), std::runtime_error);
    EXPECT_THROW(view_buffer->get_ptr(), std::runtime_error);
    EXPECT_NO_THROW(view_buffer->release());
    EXPECT_NO_THROW(view_buffer.reset());
}

TEST(FileViewBufferTest, read_file) {
    std::vector<char> test_data;
    test_data.resize(400);
    std::iota(test_data.begin(), test_data.end(), 0);

    const auto file_path = utils::generateTestFilePrefix();
    std::ofstream os(file_path, std::ios::binary);
    os.write(test_data.data(), test_data.size());
    os.close();

    std::vector<std::tuple<size_t, size_t, size_t>> test_params = {{0, 10, 64},
                                                                   {5, 20, 64},
                                                                   {50, 100, 64},
                                                                   {0, test_data.size(), 64},
                                                                   {17, 13, 29}};
    for (const auto& [offset, size, alignment] : test_params) {
        std::unique_ptr<AlignedBuffer> buffer = std::make_unique<FileViewBuffer>(file_path, offset, size, alignment);
        char* data_ptr = nullptr;
        EXPECT_NO_THROW((data_ptr = buffer->get_ptr<char>()));
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data_ptr) % alignment, 0);
        EXPECT_EQ(buffer->size(), size);
        EXPECT_EQ(std::vector<char>(data_ptr, data_ptr + size),
                  std::vector<char>(test_data.begin() + offset, test_data.begin() + offset + size));
    }
    std::filesystem::remove(file_path);
}
}  // namespace ov::test
