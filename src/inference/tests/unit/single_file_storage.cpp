// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/single_file_storage.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::test {

using runtime::SingleFileStorage;

namespace {
constexpr uint64_t version_size() {
    return 3 * sizeof(uint16_t);  // major, minor, patch
}
}  // namespace

class SingleFileStorageTest : public ::testing::Test {
protected:
    std::filesystem::path m_file_path;
    std::unique_ptr<SingleFileStorage> m_storage;

    void SetUp() override {
        m_file_path = ov::test::utils::generateTestFilePrefix() + ".bin";
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
    std::ifstream stream(m_file_path, std::ios::binary);

    std::vector<uint16_t> header_data(3);
    stream.read(reinterpret_cast<char*>(header_data.data()), 6);
    const auto last_pos = stream.tellg();
    ASSERT_NE(last_pos, std::streampos(-1));
    const auto end_pos = stream.seekg(0, std::ios::end).tellg();
    EXPECT_EQ(last_pos, end_pos);  // No more data after header in just created file

    util::Version read_version{header_data[0], header_data[1], header_data[2]};
    EXPECT_EQ(read_version, SingleFileStorage::m_version);
}

TEST_F(SingleFileStorageTest, WriteReadCacheEntry) {
    const std::vector<std::pair<std::string, std::vector<uint8_t>>> test_blobs{
        {"12", std::vector<uint8_t>(124, 0xAB)},
        {"0345", std::vector<uint8_t>(481, 0xCD)},
        {"006789", std::vector<uint8_t>(4967, 0xEF)},
        {std::to_string(UINT64_MAX), std::vector<uint8_t>(1, 0)},
    };

    for (const auto& test_blob : test_blobs) {
        const auto& blob_id = test_blob.first;
        const auto& blob_data = test_blob.second;
        m_storage->write_cache_entry(blob_id, [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }

    const auto blob_read_test = [&](SingleFileStorage& storage) {
        size_t read_count = 0;
        for (const auto& test_blob : test_blobs) {
            const auto& blob_id = test_blob.first;
            const auto& blob_data = test_blob.second;
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
                // CVS-181859 Check support for multimap memory mapping
                auto& tensor = std::get<const ov::Tensor>(compiled_blob);
                ASSERT_EQ(tensor.get_byte_size(), blob_data.size());
                std::vector<uint8_t> read_data(blob_data.size());
                std::memcpy(read_data.data(), tensor.data(), tensor.get_byte_size());
                ASSERT_EQ(blob_data, read_data);
            });
        }
        EXPECT_EQ(read_count, 2 * test_blobs.size());
    };

    blob_read_test(*m_storage);
    m_storage.reset();
    SingleFileStorage reopened_storage(m_file_path);
    blob_read_test(reopened_storage);
}

TEST_F(SingleFileStorageTest, BlobAlignment) {
    const std::unordered_map<uint64_t, std::vector<uint8_t>> test_blobs{{1, std::vector<uint8_t>(4099, 0xAB)},
                                                                        {2, std::vector<uint8_t>(400, 0xCD)},
                                                                        {3, std::vector<uint8_t>(5, 0xEF)}};
    for (const auto& test_blob : test_blobs) {
        const auto& blob_id = test_blob.first;
        const auto& blob_data = test_blob.second;
        m_storage->write_cache_entry(std::to_string(blob_id), [&](std::ostream& stream) {
            stream.write(reinterpret_cast<const char*>(blob_data.data()), blob_data.size());
        });
    }

    std::ifstream stream(m_file_path, std::ios::binary | std::ios::ate);
    const auto stream_end = stream.tellg();
    stream.seekg(version_size(), std::ios::beg);

    while (stream.good() && stream.tellg() < stream_end) {
        SingleFileStorage::Tag tag;
        runtime::TLVTraits::LengthType length;
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        ASSERT_TRUE(stream.good());
        stream.read(reinterpret_cast<char*>(&length), sizeof(length));
        ASSERT_TRUE(stream.good());
        const auto blob_id_pos = stream.tellg();
        if (tag == SingleFileStorage::Tag::Blob) {
            SingleFileStorage::BlobIdType id;
            stream.read(reinterpret_cast<char*>(&id), sizeof(id));
            ASSERT_TRUE(stream.good());
            SingleFileStorage::PadSizeType padding_size;
            stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
            ASSERT_TRUE(stream.good());

            stream.seekg(padding_size, std::ios::cur);
            const auto blob_data_pos = stream.tellg();
            EXPECT_EQ(blob_data_pos % SingleFileStorage::blob_alignment, 0)
                << "Blob with id " << id << " is not properly aligned";

            const auto expected_pos = blob_id_pos + static_cast<std::streamoff>(length);
            stream.seekg(test_blobs.at(id).size(), std::ios::cur);
            ASSERT_EQ(expected_pos, stream.tellg()) << "Blob with id " << id << " has incorrect record size";
        } else {
            stream.seekg(length, std::ios::cur);
        }
    }
}

TEST_F(SingleFileStorageTest, AppendOnlyCacheEntry) {
    const auto blob_id = std::string{"123"};
    m_storage->write_cache_entry(blob_id, [&](std::ostream& s) {
        // Although pointless it shall be harmless to write nothing
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

// ---------------------------------------------------------------------------
// Large blob test: write >= 4 MB so that the real DEFAULT_THRESHOLD (4 MB) in
// ParallelReadStreamBuf is crossed on the non-mmap read path.  This exercises
// the SingleFileStorage → ParallelReadStreamBuf integration end-to-end,
// including the blob alignment padding before the data and the non-zero
// header_offset passed to the streambuf constructor.
// ---------------------------------------------------------------------------
TEST_F(SingleFileStorageTest, WriteReadLargeBlob_ParallelPath) {
    // 5 MB + 1 byte so that even after padding the blob data itself exceeds the
    // 4 MB parallel threshold.
    constexpr size_t kBlobSize = 5UL * 1024 * 1024 + 1;
    const std::string blob_id = "999";

    // Build a deterministic pattern so byte-exact comparison is possible.
    std::vector<uint8_t> expected(kBlobSize);
    for (size_t i = 0; i < kBlobSize; ++i) {
        expected[i] = static_cast<uint8_t>(i % 251u);
    }

    m_storage->write_cache_entry(blob_id, [&](std::ostream& s) {
        s.write(reinterpret_cast<const char*>(expected.data()), static_cast<std::streamsize>(kBlobSize));
    });

    // --- non-mmap path (ParallelReadStreamBuf) ---
    m_storage->read_cache_entry(blob_id, /*enable_mmap=*/false, [&](const ICacheManager::CompiledBlobVariant& cv) {
        auto& stream = std::get<std::reference_wrapper<std::istream>>(cv).get();

        std::vector<uint8_t> got(kBlobSize);
        ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kBlobSize)));
        EXPECT_EQ(got, expected) << "Parallel read returned incorrect data for large blob";
    });

    // Reset and re-open to ensure the content survives a round-trip through
    // build_content_index (re-scans the file and rebuilds m_blob_index).
    m_storage.reset();
    SingleFileStorage reopened(m_file_path);
    reopened.read_cache_entry(blob_id, /*enable_mmap=*/false, [&](const ICacheManager::CompiledBlobVariant& cv) {
        auto& stream = std::get<std::reference_wrapper<std::istream>>(cv).get();

        std::vector<uint8_t> got(kBlobSize);
        ASSERT_TRUE(stream.read(reinterpret_cast<char*>(got.data()), static_cast<std::streamsize>(kBlobSize)));
        EXPECT_EQ(got, expected) << "Parallel read after re-open returned incorrect data for large blob";
    });
}

TEST_F(SingleFileStorageTest, ContextMetaWriteRead) {
    weight_sharing::Context test_context;
    test_context.m_weight_registry[1][11] = {100, 200, element::Type_t::f32};
    test_context.m_weight_registry[1][12] = {300, 400, element::Type_t::i8};
    test_context.m_weight_registry[2][21] = {500, 600, element::Type_t::u8};
    m_storage->write_context(test_context);

    const auto meta_read_test = [&](SingleFileStorage& storage) {
        auto got_context = storage.get_context();
        EXPECT_EQ(got_context->m_weight_registry.size(), 2);

        for (const auto& [source_id, const_meta] : test_context.m_weight_registry) {
            auto& got_meta_data = got_context->m_weight_registry;
            ASSERT_EQ(got_meta_data.count(source_id), 1);
            EXPECT_EQ(got_context->m_weight_registry[source_id].size(),
                      test_context.m_weight_registry[source_id].size());
            for (const auto& [const_id, expected_props] : const_meta) {
                ASSERT_EQ(got_meta_data[source_id].count(const_id), 1);

                auto& got_props = got_meta_data[source_id][const_id];
                EXPECT_EQ(got_props.m_offset, expected_props.m_offset);
                EXPECT_EQ(got_props.m_size, expected_props.m_size);
                EXPECT_EQ(got_props.m_type, expected_props.m_type);
            }
        }
    };

    meta_read_test(*m_storage);
    m_storage.reset();

    SingleFileStorage reopened_storage(m_file_path);
    meta_read_test(reopened_storage);
}

TEST_F(SingleFileStorageTest, ContextMetaAppendDelta) {
    weight_sharing::Context test_context;
    test_context.m_weight_registry[1][11] = {100, 200, element::Type_t::f32};
    m_storage->write_context(test_context);

    test_context.m_weight_registry[1][12] = {300, 400, element::Type_t::i8};
    test_context.m_weight_registry[2][21] = {500, 600, element::Type_t::u8};
    m_storage->write_context(test_context);

    auto got_context = m_storage->get_context();
    EXPECT_EQ(got_context->m_weight_registry.size(), 2);
    EXPECT_EQ(got_context->m_weight_registry[1].size(), 2);
    EXPECT_EQ(got_context->m_weight_registry[2].size(), 1);
    m_storage.reset();
    const auto file_size_after_first_write = test::utils::fileSize(m_file_path.string());

    SingleFileStorage{m_file_path}.write_context(test_context);
    const auto file_size_after_second_write = test::utils::fileSize(m_file_path.string());
    EXPECT_EQ(file_size_after_second_write, file_size_after_first_write)
        << "Rewriting the same context should not increase file size";
}

TEST_F(SingleFileStorageTest, ContextWeightSourceWrite) {
    weight_sharing::Context test_context;
    const auto buffer = std::make_shared<ov::AlignedBuffer>(1024);
    for (size_t i = 0; i < buffer->size(); i += 2) {
        buffer->get_ptr<uint8_t>()[i] = 0xAB;
        buffer->get_ptr<uint8_t>()[i + 1] = 0xCD;
    }
    test_context.m_cache_sources[1].m_weights = std::weak_ptr<ov::AlignedBuffer>{buffer};
    m_storage->write_context(test_context);

    std::ifstream stream(m_file_path, std::ios::binary | std::ios::ate);
    const auto stream_end = stream.tellg();
    stream.seekg(version_size(), std::ios::beg);

    while (stream.good() && stream.tellg() < stream_end) {
        SingleFileStorage::Tag tag;
        runtime::TLVTraits::LengthType length;
        stream.read(reinterpret_cast<char*>(&tag), sizeof(tag));
        ASSERT_TRUE(stream.good());
        stream.read(reinterpret_cast<char*>(&length), sizeof(length));
        ASSERT_TRUE(stream.good());
        if (tag == SingleFileStorage::Tag::WeightSource) {
            SingleFileStorage::DataIdType device_id, source_id;
            SingleFileStorage::PadSizeType padding_size;
            stream.read(reinterpret_cast<char*>(&device_id), sizeof(device_id));
            stream.read(reinterpret_cast<char*>(&source_id), sizeof(source_id));
            stream.read(reinterpret_cast<char*>(&padding_size), sizeof(padding_size));
            ASSERT_TRUE(stream.good());

            stream.seekg(padding_size, std::ios::cur);
            const auto weight_pos = stream.tellg();
            ASSERT_EQ(weight_pos % SingleFileStorage::blob_alignment, 0);
            const auto weight_size =
                length - sizeof(device_id) - sizeof(source_id) - sizeof(padding_size) - padding_size;
            ASSERT_EQ(weight_size, buffer->size());
            std::vector<uint8_t> read_data(weight_size);
            stream.read(reinterpret_cast<char*>(read_data.data()), read_data.size());
            EXPECT_EQ(std::memcmp(read_data.data(), buffer->get_ptr(), buffer->size()), 0);
        } else {
            stream.seekg(length, std::ios::cur);
        }
    }
}

TEST_F(SingleFileStorageTest, ContextWeightSourceAppendDelta) {
    weight_sharing::Context test_context;
    const auto buffer_1 = std::make_shared<ov::AlignedBuffer>(1024);
    test_context.m_cache_sources[1].m_weights = buffer_1;
    m_storage->write_context(test_context);

    const auto buffer_2 = std::make_shared<ov::AlignedBuffer>(47);
    test_context.m_cache_sources[11].m_weights = buffer_2;
    m_storage->write_context(test_context);

    EXPECT_EQ(m_storage->get_context()->m_cache_sources.size(), 2);
    m_storage.reset();
    const auto file_size_after_first_write = test::utils::fileSize(m_file_path.string());

    SingleFileStorage{m_file_path}.write_context(test_context);
    const auto file_size_after_second_write = test::utils::fileSize(m_file_path.string());
    EXPECT_EQ(file_size_after_second_write, file_size_after_first_write)
        << "Rewriting the same context should not increase file size";
}
}  // namespace ov::test
