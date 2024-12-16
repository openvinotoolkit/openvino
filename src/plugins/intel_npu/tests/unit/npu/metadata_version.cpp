// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"

using namespace intel_npu;

using MetadataUnitTests = ::testing::Test;

struct MetadataTest : Metadata<CURRENT_METADATA_VERSION> {
    void set_version(uint32_t newVersion) {
        _version = newVersion;
    }

    void set_ov_version(const OpenvinoVersion& newVersion) {
        _ovVersion = newVersion;
    }
};

TEST_F(MetadataUnitTests, readUnversionedBlob) {
    std::vector<uint8_t> blob(50, 68);

    std::unique_ptr<MetadataBase> storedMeta;
    ASSERT_ANY_THROW(storedMeta = read_metadata_from(blob));
    ASSERT_EQ(storedMeta, nullptr);
}

TEST_F(MetadataUnitTests, writeAndReadMetadataFromBlob) {
    std::stringstream stream;
    size_t blobSize = 0;
    auto meta = MetadataTest();

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    blobSize = stream.str().length();

    std::vector<uint8_t> blob(blobSize);
    OV_ASSERT_NO_THROW(stream.read(reinterpret_cast<char*>(blob.data()), blobSize));
    auto storedMeta = read_metadata_from(blob);
    ASSERT_NE(storedMeta, nullptr);
    ASSERT_TRUE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidOpenvinoVersion) {
    size_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest();

    OpenvinoVersion badOvVersion("just_some_wrong_ov_version");
    meta.set_ov_version(badOvVersion);

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    blobSize = stream.str().length();

    std::vector<uint8_t> blob(blobSize);
    OV_ASSERT_NO_THROW(stream.read(reinterpret_cast<char*>(blob.data()), blobSize));
    auto storedMeta = read_metadata_from(blob);
    ASSERT_NE(storedMeta, nullptr);
    ASSERT_FALSE(storedMeta->is_compatible());
}

TEST_F(MetadataUnitTests, writeAndReadInvalidMetadataVersion) {
    size_t blobSize = 0;
    std::stringstream stream;
    auto meta = MetadataTest();

    constexpr uint32_t dummy_version = make_version(0x00007E57, 0x0000AC3D);
    meta.set_version(dummy_version);

    OV_ASSERT_NO_THROW(meta.write(stream));
    OV_ASSERT_NO_THROW(stream.write(reinterpret_cast<const char*>(&blobSize), sizeof(blobSize)));
    OV_ASSERT_NO_THROW(stream.write(MAGIC_BYTES.data(), MAGIC_BYTES.size()));

    blobSize = stream.str().length();

    std::vector<uint8_t> blob(blobSize);
    OV_ASSERT_NO_THROW(stream.read(reinterpret_cast<char*>(blob.data()), blobSize));
    ASSERT_ANY_THROW(auto storedMeta = read_metadata_from(blob));
}
