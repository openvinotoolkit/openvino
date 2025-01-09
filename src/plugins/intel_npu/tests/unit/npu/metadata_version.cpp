// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"
#include "openvino/core/version.hpp"

using namespace intel_npu;

using MetadataUnitTests = ::testing::Test;

struct MetadataTest : Metadata<CURRENT_METADATA_VERSION> {
    MetadataTest(uint64_t blobSize, std::optional<std::string_view> ovVersion)
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
    auto meta = MetadataTest(blobSize, ov::get_openvino_version().buildNumber);

    OV_ASSERT_NO_THROW(meta.write(stream));

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    ASSERT_TRUE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidOpenvinoVersion) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest(blobSize, "just_some_wrong_ov_version");

    OV_ASSERT_NO_THROW(meta.write(stream));

    std::unique_ptr<MetadataBase> storedMeta;
    OV_ASSERT_NO_THROW(storedMeta = read_metadata_from(stream));
    ASSERT_FALSE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidMetadataVersion) {
    uint64_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest(blobSize, std::nullopt);

    constexpr uint32_t dummy_version = MetadataBase::make_version(0x00007E57, 0x0000AC3D);
    meta.set_version(dummy_version);

    OV_ASSERT_NO_THROW(meta.write(stream));
    ASSERT_ANY_THROW(auto storedMeta = read_metadata_from(stream));
}

struct MetadataVersionTestFixture : Metadata<CURRENT_METADATA_VERSION>, ::testing::TestWithParam<uint32_t> {
public:
    std::stringstream blob;

    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }

    MetadataVersionTestFixture() : Metadata<CURRENT_METADATA_VERSION>(0, std::nullopt) {}

    MetadataVersionTestFixture(uint64_t blobSize, std::optional<std::string_view> ovVersion)
        : Metadata<CURRENT_METADATA_VERSION>(blobSize, ovVersion) {}

    void TestBody() override {}
};

TEST_P(MetadataVersionTestFixture, writeAndReadInvalidMetadataVersion) {
    MetadataVersionTestFixture dummyMeta = MetadataVersionTestFixture(0, ov::get_openvino_version().buildNumber);
    uint32_t metaVersion = GetParam();

    dummyMeta.set_version(metaVersion);

    OV_ASSERT_NO_THROW(dummyMeta.write(blob));
    ASSERT_ANY_THROW(read_metadata_from(blob));
}

constexpr uint16_t currentMajor = MetadataBase::get_major(CURRENT_METADATA_VERSION),
                   currentMinor = MetadataBase::get_minor(CURRENT_METADATA_VERSION);

INSTANTIATE_TEST_CASE_P(MetadataUnitTests,
                        MetadataVersionTestFixture,
                        ::testing::Values(MetadataBase::make_version(currentMajor, currentMinor + 1),
                                          MetadataBase::make_version(currentMajor, currentMinor - 1),
                                          MetadataBase::make_version(currentMajor + 1, currentMinor),
                                          MetadataBase::make_version(currentMajor + 1, currentMinor + 1),
                                          MetadataBase::make_version(currentMajor + 1, currentMinor - 1),
                                          MetadataBase::make_version(currentMajor - 1, currentMinor),
                                          MetadataBase::make_version(currentMajor - 1, currentMinor + 1),
                                          MetadataBase::make_version(currentMajor - 1, currentMinor - 1)));
