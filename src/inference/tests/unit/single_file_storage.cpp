// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../src/single_file_storage.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"

namespace ov::test {

class SingleFileStorageTest : public ::testing::Test {
protected:
    std::filesystem::path m_cache_file_path;

    void SetUp() override {
        m_cache_file_path = ov::test::utils::generateTestFilePrefix() + ".cache";
    }

    void TearDown() override {
        std::filesystem::remove(m_cache_file_path);
    }
};

TEST_F(SingleFileStorageTest, WriteReadCacheEntry) {
    SingleFileStorage storage(m_cache_file_path);
    const std::vector<std::pair<std::string, std::vector<uint8_t>>> test_blobs{
        {"blob1", std::vector<uint8_t>(1024, 0xAB)},
        // {"blob2", std::vector<uint8_t>(2048, 0xCD)},
        // {"blob3", std::vector<uint8_t>(4096, 0xEF)},
    };

    for (const auto& [blob_id, blob_data] : test_blobs) {
        storage.write_cache_entry(blob_id, [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }

    for (const auto& [blob_id, blob_data] : test_blobs) {
        storage.read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
            ASSERT_TRUE(std::holds_alternative<std::reference_wrapper<std::istream>>(compiled_blob));
            auto& stream = std::get<std::reference_wrapper<std::istream>>(compiled_blob).get();
            std::vector<uint8_t> read_data(blob_data.size());
            stream.read(reinterpret_cast<char*>(read_data.data()), read_data.size());
            ASSERT_EQ(blob_data, read_data);
        });

        storage.read_cache_entry(blob_id, true, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
            ASSERT_TRUE(std::holds_alternative<const ov::Tensor>(compiled_blob));
            auto& tensor = std::get<const ov::Tensor>(compiled_blob);
            ASSERT_EQ(tensor.get_byte_size(), blob_data.size());
            std::vector<uint8_t> read_data(blob_data.size());
            std::memcpy(read_data.data(), tensor.data(), tensor.get_byte_size());
            ASSERT_EQ(blob_data, read_data);
        });
    }
}
}  // namespace ov::test
