// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"
#include "openvino/core/version.hpp"

using namespace intel_npu;

using MetadataUnitTests = ::testing::Test;

struct MetadataTest : Metadata<CURRENT_METADATA_VERSION> {
    MetadataTest(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion)
        : Metadata<CURRENT_METADATA_VERSION>(blobSize, ovVersion) {}

    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }
};

TEST_F(MetadataUnitTests, readUnversionedBlob) {
    std::stringstream blob("this_is an_unversioned bl0b");

    std::unique_ptr<MetadataBase> storedMeta;
    ASSERT_ANY_THROW(storedMeta = read_metadata_from(blob));
}

TEST_F(MetadataUnitTests, writeAndReadCurrentMetadataFromBlob) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest(blobSize, CURRENT_OPENVINO_VERSION);

    OV_ASSERT_NO_THROW(meta.write(stream));

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    ASSERT_TRUE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidOpenvinoVersion) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    OpenvinoVersion wrongOvVersion(0xF0C, 0xACC, 0x1A);
    auto meta = MetadataTest(blobSize, wrongOvVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    ASSERT_FALSE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidMetadataVersion) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest(blobSize, std::nullopt);

    constexpr uint32_t dummyVersion = MetadataBase::make_version(0x00007E57, 0x0000AC3D);
    meta.set_version(dummyVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));
    ASSERT_ANY_THROW(auto storedMeta = read_metadata_from(stream));
}

TEST_F(MetadataUnitTests, writeAndReadMetadataWithNewerMinorVersion) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest(blobSize, CURRENT_OPENVINO_VERSION);

    constexpr uint32_t dummyVersion =
        MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION, CURRENT_METADATA_MINOR_VERSION + 1);
    meta.set_version(dummyVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));
    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    storedMeta->is_compatible();
}

struct MetadataVersionTestFixture : Metadata<CURRENT_METADATA_VERSION>, ::testing::TestWithParam<uint32_t> {
public:
    std::stringstream blob;

    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }

    MetadataVersionTestFixture() : Metadata<CURRENT_METADATA_VERSION>(0, std::nullopt) {}

    MetadataVersionTestFixture(uint64_t blobSize, std::optional<OpenvinoVersion> ovVersion)
        : Metadata<CURRENT_METADATA_VERSION>(blobSize, ovVersion) {}

    void TestBody() override {}

    static std::string getTestCaseName(testing::TestParamInfo<MetadataVersionTestFixture::ParamType> info);
};

std::string MetadataVersionTestFixture::getTestCaseName(
    testing::TestParamInfo<MetadataVersionTestFixture::ParamType> info) {
    std::ostringstream result;
    result << "major version=" << MetadataBase::get_major(info.param)
           << ", minor version=" << MetadataBase::get_minor(info.param);
    return result.str();
}

TEST_P(MetadataVersionTestFixture, writeAndReadInvalidMetadataVersion) {
    uint32_t metaVersion = GetParam();
    if (CURRENT_METADATA_MAJOR_VERSION == MetadataBase::get_major(metaVersion) && CURRENT_METADATA_MINOR_VERSION == 0) {
        GTEST_SKIP() << "Skipping single test since there is no case of lower minor version than actual.";
    }

    OpenvinoVersion dummyOvVersion(0x5A, 0x1A, 0xD);
    MetadataVersionTestFixture dummyMeta = MetadataVersionTestFixture(0, dummyOvVersion);
    dummyMeta.set_version(metaVersion);

    OV_ASSERT_NO_THROW(dummyMeta.write(blob));
    EXPECT_ANY_THROW(read_metadata_from(blob));
    ASSERT_FALSE(dummyMeta.is_compatible());
}

const std::vector badMetadataVersions = {
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION, CURRENT_METADATA_MINOR_VERSION - 1),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION + 1, CURRENT_METADATA_MINOR_VERSION),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION + 1, CURRENT_METADATA_MINOR_VERSION + 1),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION + 1, CURRENT_METADATA_MINOR_VERSION - 1),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION - 1, CURRENT_METADATA_MINOR_VERSION),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION - 1, CURRENT_METADATA_MINOR_VERSION + 1),
    MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION - 1, CURRENT_METADATA_MINOR_VERSION - 1)};

INSTANTIATE_TEST_SUITE_P(MetadataUnitTests,
                         MetadataVersionTestFixture,
                         ::testing::ValuesIn(badMetadataVersions),
                         MetadataVersionTestFixture::getTestCaseName);

TEST_F(MetadataUnitTests, writeAndReadMetadataWithNewerFieldAtEnd) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    OpenvinoVersion dummyOvVersion(0x0FF, 0xC0FF, 0xEEEE);
    auto meta = MetadataTest(blobSize, dummyOvVersion);

    constexpr uint32_t dummyVersion =
        MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION, CURRENT_METADATA_MINOR_VERSION + 1);
    meta.set_version(dummyVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));

    // inserting a new field at the end of the blob, between last metadata field and blobDataSize
    std::string temp = stream.str();
    size_t offset = MAGIC_BYTES.size() + sizeof(uint64_t);
    temp.insert(temp.length() - offset, "new metadata field");
    stream.str("");
    stream << temp;

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    ASSERT_FALSE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadMetadataWithNewerFieldAtMiddle) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    OpenvinoVersion dummyOvVersion(0xFA, 0x1A, 0xFE1);
    auto meta = MetadataTest(blobSize, dummyOvVersion);

    constexpr uint32_t dummyVersion =
        MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION + 1, CURRENT_METADATA_MINOR_VERSION);
    meta.set_version(dummyVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));

    // inserting a new field at the middle of the blob, between metadata version and OV version size
    std::string temp = stream.str();
    size_t offset = sizeof(CURRENT_METADATA_VERSION);
    temp.insert(offset, "new metadata field");
    stream.str("");
    stream << temp;

    std::unique_ptr<MetadataBase> storedMeta;
    EXPECT_ANY_THROW(storedMeta = read_metadata_from(stream));
}

TEST_F(MetadataUnitTests, writeAndReadMetadataWithRemovedField) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    OpenvinoVersion dummyOvVersion(0xBA, 0xB1, 0xC);
    auto meta = MetadataTest(blobSize, dummyOvVersion);

    constexpr uint32_t dummyVersion =
        MetadataBase::make_version(CURRENT_METADATA_MAJOR_VERSION + 1, CURRENT_METADATA_MINOR_VERSION);
    meta.set_version(dummyVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));

    // removing fields between metadata version and blob data size
    std::string temp = stream.str();
    size_t offset = sizeof(CURRENT_METADATA_VERSION), size = offset + MAGIC_BYTES.size() + sizeof(uint64_t);
    temp.replace(offset, temp.length() - size, "");
    stream.str("");
    stream << temp;

    std::unique_ptr<MetadataBase> storedMeta;
    EXPECT_ANY_THROW(storedMeta = read_metadata_from(stream));
}
