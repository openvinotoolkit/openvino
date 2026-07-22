// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cache_manager.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <streambuf>
#include <string>

#include "common_test_utils/common_utils.hpp"

namespace ov::test {
namespace {

std::string read_blob(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class FlushFailingStreambuf final : public std::streambuf {
public:
    explicit FlushFailingStreambuf(std::streambuf* delegate) : m_delegate(delegate) {}

protected:
    std::streamsize xsputn(const char* s, std::streamsize count) override {
        return m_delegate->sputn(s, count);
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        return m_delegate->sputc(traits_type::to_char_type(ch));
    }

    int sync() override {
        return -1;
    }

private:
    std::streambuf* m_delegate;
};

class FileStorageCacheManagerTest : public ::testing::Test {
protected:
    std::filesystem::path m_cache_dir;
    std::unique_ptr<ICacheManager> m_cache_manager;

    void SetUp() override {
        m_cache_dir = ov::test::utils::generateTestFilePrefix();
        m_cache_manager = std::make_unique<FileStorageCacheManager>(m_cache_dir);
    }

    void TearDown() override {
        m_cache_manager.reset();
        std::filesystem::remove_all(m_cache_dir);
    }

    std::filesystem::path blob_path(const std::string& id) const {
        return m_cache_dir / (id + ".blob");
    }
};

TEST_F(FileStorageCacheManagerTest, RemovesBlobAfterWriteFailureWhenCallerCleansUp) {
    EXPECT_THROW(m_cache_manager->write_cache_entry("1",
                                                    [&](std::ostream& stream) {
                                                        stream << "partial";
                                                        throw std::runtime_error("write failed");
                                                    }),
                 std::runtime_error);

    EXPECT_NO_THROW(m_cache_manager->remove_cache_entry("1"));
    EXPECT_FALSE(std::filesystem::exists(blob_path("1")));
}

TEST_F(FileStorageCacheManagerTest, RemovesBlobAfterStreamFailureWhenCallerCleansUp) {
    EXPECT_THROW(m_cache_manager->write_cache_entry("2",
                                                    [&](std::ostream& stream) {
                                                        stream << "partial";
                                                        stream.setstate(std::ios_base::badbit);
                                                    }),
                 std::ios_base::failure);

    EXPECT_NO_THROW(m_cache_manager->remove_cache_entry("2"));
    EXPECT_FALSE(std::filesystem::exists(blob_path("2")));
}

TEST_F(FileStorageCacheManagerTest, OverwritesExistingBlobWhenEntryAlreadyExists) {
    m_cache_manager->write_cache_entry("7", [&](std::ostream& stream) {
        stream << "cached";
    });

    bool writer_called = false;
    EXPECT_NO_THROW(m_cache_manager->write_cache_entry("7", [&](std::ostream& stream) {
        writer_called = true;
        stream << "new";
    }));

    EXPECT_TRUE(writer_called);
    EXPECT_EQ(read_blob(blob_path("7")), "new");
}

TEST_F(FileStorageCacheManagerTest, DoesNotLeaveAuxiliaryFilesInCacheDirectory) {
    m_cache_manager->write_cache_entry("8", [&](std::ostream& stream) {
        stream << "cached";
    });

    std::vector<std::filesystem::path> entries;
    for (const auto& entry : std::filesystem::directory_iterator(m_cache_dir)) {
        entries.push_back(entry.path().filename());
    }

    ASSERT_EQ(entries.size(), 1);
    EXPECT_EQ(entries.front(), std::filesystem::path{"8.blob"});
}

}  // namespace
}  // namespace ov::test
