// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
#include "metadata.hpp"

using namespace intel_npu;

namespace {

constexpr std::string_view DUMMY_COMPILER_PAYLOAD = "1";
constexpr std::string_view RAW_BLOB = DUMMY_COMPILER_PAYLOAD;
constexpr size_t MINIMUM_BLOB_SIZE = sizeof(uint32_t) + sizeof(uint64_t) + MAGIC_BYTES.size();

std::string build_blob_format_v1_with_magic() {
    std::ostringstream stream;
    stream << DUMMY_COMPILER_PAYLOAD.data();
    Metadata<CURRENT_METADATA_VERSION> metadata(DUMMY_COMPILER_PAYLOAD.size());
    metadata.write(stream);

    const std::string blob = stream.str();
    OPENVINO_ASSERT(blob.size() >= MINIMUM_BLOB_SIZE);
    return blob;
}

std::string build_blob_format_v1_without_magic() {
    const std::string blob = build_blob_format_v1_with_magic();
    return blob.substr(0, blob.size() - MAGIC_BYTES.size());
}

}  // namespace

using testing::_;

struct BlobFormatImportersTest : public ::testing::Test {
    BlobFormatImportersTest() : config(std::make_shared<OptionsDesc>()) {}

    FilteredConfig config;
};

TEST_F(BlobFormatImportersTest, FactoryEmptyInputFails) {
    std::istringstream input_stream("");
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_stream, false, nullptr, config), ov::Exception, _);
    const ov::Tensor input_tensor;
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_tensor, false, nullptr, config), ov::Exception, _);
}

TEST_F(BlobFormatImportersTest, FactoryNoMagicNoRawFails) {
    const std::string blob = build_blob_format_v1_without_magic();
    std::istringstream input_stream(blob);
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_stream, false, nullptr, config), ov::Exception, _);
    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    OV_EXPECT_THROW(blob_format_importer_factory::create(input_tensor, false, nullptr, config), ov::Exception, _);
}

TEST_F(BlobFormatImportersTest, FactoryNoMagicRawPasses) {
    const std::string blob(RAW_BLOB);
    std::istringstream input_stream(blob);
    std::unique_ptr<IBlobFormatImporter> importer;

    OV_ASSERT_NO_THROW(importer = blob_format_importer_factory::create(input_stream, true, nullptr, config));
    ASSERT_TRUE(dynamic_cast<RawBlobImporter*>(importer.get()));

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(input_tensor, true, nullptr, config));
    ASSERT_TRUE(dynamic_cast<RawBlobImporter*>(importer.get()));
}

TEST_F(BlobFormatImportersTest, FactoryCorrectImporterBasedOnRaw) {
    const std::string blob = build_blob_format_v1_with_magic();
    std::istringstream input_stream(blob);
    std::unique_ptr<IBlobFormatImporter> importer;

    OV_ASSERT_NO_THROW(importer = blob_format_importer_factory::create(input_stream, true, nullptr, config));
    ASSERT_TRUE(dynamic_cast<RawBlobImporter*>(importer.get()));

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(input_tensor, true, nullptr, config));
    ASSERT_TRUE(dynamic_cast<RawBlobImporter*>(importer.get()));

    input_stream.seekg(0, std::ios::beg);
    OV_ASSERT_NO_THROW(importer = blob_format_importer_factory::create(input_stream, false, nullptr, config));
    ASSERT_TRUE(dynamic_cast<BlobFormatV1Importer*>(importer.get()));

    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(input_tensor, false, nullptr, config));
    ASSERT_TRUE(dynamic_cast<BlobFormatV1Importer*>(importer.get()));
}
