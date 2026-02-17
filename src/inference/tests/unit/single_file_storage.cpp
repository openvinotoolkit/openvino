// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../src/single_file_storage.hpp"

#include <gtest/gtest.h>

#include "../../src/storage_codecs.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

namespace ov::test {

class SingleFileStorageTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;
    std::unique_ptr<SingleFileStorage> m_storage;

    void SetUp() override {
        m_file_path = ov::test::utils::generateTestFilePrefix() + ".cache";
        m_storage = std::make_unique<SingleFileStorage>(m_file_path);
        ASSERT_TRUE(std::filesystem::exists(m_file_path));
    }

    void TearDown() override {
        m_storage.reset();
        std::filesystem::remove(m_file_path);
    }
};

TEST_F(SingleFileStorageTest, FileHeader) {
    std::ifstream stream(m_file_path, std::ios_base::binary);
    SingleFileStorageHeaderCodec header{};
    stream >> header;

    char c;
    stream.get(c);
    EXPECT_EQ(c, '\0');
    EXPECT_TRUE(stream.eof());  // No more data after header in just created file

    EXPECT_EQ(header.major_version, SingleFileStorage::major_version);
    EXPECT_EQ(header.minor_version, SingleFileStorage::minor_version);
    EXPECT_EQ(header.weight_path, "weight/path");
}

TEST_F(SingleFileStorageTest, WriteReadCacheEntry) {
    const std::vector<std::pair<std::string, std::vector<uint8_t>>> test_blobs{
        {"blob 1", std::vector<uint8_t>(124, 0xAB)},
        {"blob 2", std::vector<uint8_t>(481, 0xCD)},
        {"blob 3", std::vector<uint8_t>(4967, 0xEF)},
    };

    for (const auto& [blob_id, blob_data] : test_blobs) {
        m_storage->write_cache_entry(blob_id, [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }
    m_storage.reset();

    size_t expected_read_count = 0;
    SingleFileStorage storage(m_file_path);
    for (const auto& [blob_id, blob_data] : test_blobs) {
        storage.read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
            ++expected_read_count;
            ASSERT_TRUE(std::holds_alternative<std::reference_wrapper<std::istream>>(compiled_blob));
            auto& stream = std::get<std::reference_wrapper<std::istream>>(compiled_blob).get();
            std::vector<uint8_t> read_data(blob_data.size());
            stream.read(reinterpret_cast<char*>(read_data.data()), read_data.size());
            EXPECT_EQ(blob_data, read_data);
        });
    }
    EXPECT_EQ(expected_read_count, test_blobs.size());

    // no support for multimap memory mapping yet
    // storage.read_cache_entry(blob_id, true, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
    //     ASSERT_TRUE(std::holds_alternative<const ov::Tensor>(compiled_blob));
    //     auto& tensor = std::get<const ov::Tensor>(compiled_blob);
    //     ASSERT_EQ(tensor.get_byte_size(), blob_data.size());
    //     std::vector<uint8_t> read_data(blob_data.size());
    //     std::memcpy(read_data.data(), tensor.data(), tensor.get_byte_size());
    //     ASSERT_EQ(blob_data, read_data);
    // });
}

TEST_F(SingleFileStorageTest, AppendOnly) {
    const auto blob_id = std::string{"blob id"};
    m_storage->write_cache_entry(blob_id, [&](std::ostream&) {});
    OV_EXPECT_THROW_HAS_SUBSTRING(m_storage->write_cache_entry(blob_id, [&](std::ostream&) {}),
                                  ov::AssertFailure,
                                  blob_id + " already exists in cache");

    EXPECT_NO_THROW(m_storage->remove_cache_entry(blob_id));  // removal does nothing
    bool read_called = false;
    EXPECT_NO_THROW(m_storage->read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant&) {
        read_called = true;
    }));
    EXPECT_TRUE(read_called);

    EXPECT_NO_THROW(m_storage->read_cache_entry("dummy id", false, [](const ICacheManager::CompiledBlobVariant&) {
        throw "Unexpected read for dummy id";
    }));
    EXPECT_NO_THROW(m_storage->remove_cache_entry("dummy id"));
}

// TEST_F(SingleFileStorageTest, SharedContext__) {
//     void write_context_entry(const SharedContext& context) override;
//     SharedContext get_shared_context() const override;
// }
}  // namespace ov::test
