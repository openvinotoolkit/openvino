// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../src/single_file_storage.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_assertions.hpp"

namespace ov::test {

namespace {
constexpr uint64_t version_size() {
    return sizeof(TLVStorage::Version::major) + sizeof(TLVStorage::Version::minor) + sizeof(TLVStorage::Version::patch);
}
}  // namespace

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
    m_storage.reset();
    std::ifstream stream(m_file_path, std::ios_base::binary);

    std::vector<char> header_data(version_size());
    stream.read(header_data.data(), header_data.size());
    const auto last_pos = stream.tellg();
    ASSERT_NE(last_pos, std::streampos(-1));
    stream.seekg(0, std::ios_base::end);
    const auto end_pos = stream.tellg();
    EXPECT_EQ(last_pos, end_pos);  // No more data after header in just created file

    const auto major_p = reinterpret_cast<const uint16_t*>(header_data.data());
    const auto minor_p = major_p + sizeof(TLVStorage::Version::major) / sizeof(uint16_t);
    const auto patch_p = minor_p + sizeof(TLVStorage::Version::minor) / sizeof(uint16_t);
    TLVStorage::Version read_version{*major_p, *minor_p, *patch_p};
    EXPECT_EQ(read_version, SingleFileStorage::m_version);
}

TEST_F(SingleFileStorageTest, WriteReadCacheEntry) {
    const std::vector<std::pair<std::string, std::vector<uint8_t>>> test_blobs{
        {"12", std::vector<uint8_t>(124, 0xAB)},
        {"0345", std::vector<uint8_t>(481, 0xCD)},
        {"006789", std::vector<uint8_t>(4967, 0xEF)},
        {std::to_string(std::numeric_limits<uint64_t>::max()), std::vector<uint8_t>(1, 0)},
    };

    for (const auto& [blob_id, blob_data] : test_blobs) {
        m_storage->write_cache_entry(blob_id, [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }

    const auto test_storage_read = [&](SingleFileStorage& storage) {
        size_t read_count = 0;
        for (const auto& [blob_id, blob_data] : test_blobs) {
            storage.read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
                ASSERT_TRUE(std::holds_alternative<std::reference_wrapper<std::istream>>(compiled_blob));
                ++read_count;
                auto& stream = std::get<std::reference_wrapper<std::istream>>(compiled_blob).get();
                std::vector<uint8_t> read_data(blob_data.size());
                stream.read(reinterpret_cast<char*>(read_data.data()), read_data.size());
                EXPECT_EQ(blob_data, read_data);
            });

            storage.read_cache_entry(blob_id, true, [&](const ICacheManager::CompiledBlobVariant& compiled_blob) {
                ASSERT_TRUE(std::holds_alternative<const ov::Tensor>(compiled_blob));
                ++read_count;
                // todo Check support for multimap memory mapping
                // auto& tensor = std::get<const ov::Tensor>(compiled_blob);
                // ASSERT_EQ(tensor.get_byte_size(), blob_data.size());
                // std::vector<uint8_t> read_data(blob_data.size());
                // std::memcpy(read_data.data(), tensor.data(), tensor.get_byte_size());
                // ASSERT_EQ(blob_data, read_data);
            });
        }
        EXPECT_EQ(read_count, 2 * test_blobs.size());
    };

    test_storage_read(*m_storage);
    m_storage.reset();
    SingleFileStorage reopened_storage(m_file_path);
    test_storage_read(reopened_storage);
}

TEST_F(SingleFileStorageTest, Alignement) {
    const std::unordered_map<uint64_t, std::vector<uint8_t>> test_blobs{{1, std::vector<uint8_t>(4099, 0xAB)},
                                                                        {2, std::vector<uint8_t>(400, 0xCD)},
                                                                        {3, std::vector<uint8_t>(5, 0xEF)}};
    for (const auto& [blob_id, blob_data] : test_blobs) {
        m_storage->write_cache_entry(std::to_string(blob_id), [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }

    std::ifstream stream(m_file_path, std::ios_base::binary);
    stream.seekg(0, std::ios::end);
    const auto stream_end = stream.tellg();
    stream.seekg(version_size(), std::ios_base::beg);

    // todo Make it configurable and/or detect actual file system page size
    constexpr std::streamoff alignment = 4096;

    while (stream.good() && stream.tellg() < stream_end) {
        TLVStorage::Tag tag{};
        TLVStorage::length_type entry_size{};
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        ASSERT_TRUE(stream.good());
        stream.read(reinterpret_cast<char*>(&entry_size), sizeof(entry_size));
        ASSERT_TRUE(stream.good());
        const auto blob_id_pos = stream.tellg();
        if (tag == TLVStorage::Tag::Blob) {
            TLVStorage::blob_id_type id;
            stream.read(reinterpret_cast<char*>(&id), sizeof(id));
            ASSERT_TRUE(stream.good());
            TLVStorage::pad_size_type padding_size{};
            stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
            ASSERT_TRUE(stream.good());

            stream.seekg(padding_size, std::ios_base::cur);
            const auto blob_data_pos = stream.tellg();
            EXPECT_EQ(blob_data_pos % alignment, 0) << "Blob with id " << id << " is not properly aligned";

            const auto expected_pos = blob_id_pos + static_cast<std::streamoff>(entry_size);
            stream.seekg(test_blobs.at(id).size(), std::ios_base::cur);
            ASSERT_EQ(expected_pos, stream.tellg()) << "Blob with id " << id << " has incorrect entry size";
        } else {
            stream.seekg(entry_size, std::ios_base::cur);
        }
    }
}

TEST_F(SingleFileStorageTest, AppendOnly) {
    const auto blob_id = std::string{"123"};
    m_storage->write_cache_entry(blob_id, [&](std::ostream& s) {
        // Although pointless it shell be harmless to write nothing
    });
    EXPECT_NO_THROW(m_storage->remove_cache_entry(blob_id));  // removal does nothing => expect legit read
    bool read_called = false;
    EXPECT_NO_THROW(m_storage->read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant&) {
        read_called = true;
    }));
    EXPECT_TRUE(read_called);

    OV_EXPECT_THROW_HAS_SUBSTRING(m_storage->write_cache_entry(blob_id, [&](std::ostream&) {}),
                                  ov::AssertFailure,
                                  blob_id + " already exists in cache");
    m_storage.reset();

    SingleFileStorage reopened_storage(m_file_path);
    OV_EXPECT_THROW_HAS_SUBSTRING(reopened_storage.write_cache_entry(blob_id, [&](std::ostream&) {}),
                                  ov::AssertFailure,
                                  blob_id + " already exists in cache");

    EXPECT_NO_THROW(reopened_storage.read_cache_entry("987", false, [](const ICacheManager::CompiledBlobVariant&) {
        throw "Unexpected read for not stored blob id";
    }));
    EXPECT_NO_THROW(reopened_storage.remove_cache_entry("987")) << "Removal of non-existing blob id should be no-op";

    read_called = false;
    EXPECT_NO_THROW(reopened_storage.read_cache_entry(blob_id, false, [&](const ICacheManager::CompiledBlobVariant&) {
        read_called = true;
    }));
    EXPECT_TRUE(read_called);
}

// TEST_F(SingleFileStorageTest, SharedContext__) {
//     void write_context_entry(const SharedContext& context) override;
//     SharedContext get_shared_context() const override;
// }
}  // namespace ov::test
