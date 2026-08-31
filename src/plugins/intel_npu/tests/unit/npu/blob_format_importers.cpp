// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/config/options.hpp"
#include "metadata.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

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

std::string build_blob_format_v1_with_blob_type(const BlobType blob_type) {
    std::ostringstream stream;
    stream << DUMMY_COMPILER_PAYLOAD.data();
    Metadata<CURRENT_METADATA_VERSION> metadata(DUMMY_COMPILER_PAYLOAD.size(),
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                std::nullopt,
                                                blob_type);
    metadata.write(stream);

    return stream.str();
}

std::shared_ptr<ov::Model> create_simple_model() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "Simple with weights");
}

}  // namespace

using testing::_;

struct BlobFormatImportersTest : public ::testing::Test {
    BlobFormatImportersTest() : options(std::make_shared<OptionsDesc>()), config(options) {
        options->add<ALLOW_DYNAMIC_BLOB_IMPORT>();
        config.enableAll();
    }

    std::shared_ptr<OptionsDesc> options;
    std::unique_ptr<IBlobFormatImporter> importer;
    FilteredConfig config;
};

/**
 * @brief Empty blobs should not be accepted by the importer factory
 */
TEST_F(BlobFormatImportersTest, FactoryEmptyInputFails) {
    const std::string empty_buffer("");
    std::istringstream input_stream(empty_buffer);
    BlobSource source(input_stream);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({0}), empty_buffer.data());
    source = BlobSource(input_tensor);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);
}

/**
 * @brief A non-raw tensor with 1..(MAGIC_BYTES.size()-1) bytes must be rejected without reading OOB.
 * The unsigned subtraction get_byte_size() - MAGIC_BYTES.size() underflows otherwise.
 */
TEST_F(BlobFormatImportersTest, FactoryTensorSmallerThanMagicFails) {
    ASSERT_GT(MAGIC_BYTES.size(), 1u);
    for (size_t sz = 1; sz < MAGIC_BYTES.size(); ++sz) {
        const std::string blob(sz, '\x00');
        const ov::Tensor input_tensor(ov::element::Type_t::u8,
                                      ov::Shape({blob.size()}),
                                      const_cast<char*>(blob.data()));
        BlobSource source(input_tensor);
        OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);
    }
}

/**
 * @brief Non-raw blobs must contain the magic bytes
 */
TEST_F(BlobFormatImportersTest, FactoryNoMagicNoRawFails) {
    const std::string blob = build_blob_format_v1_without_magic();

    std::istringstream input_stream(blob);
    BlobSource source(input_stream);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    source = BlobSource(input_tensor);
    OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);
}

/**
 * @brief Only raw blobs can be created when the magic bytes are missing
 */
TEST_F(BlobFormatImportersTest, FactoryNoMagicRawPasses) {
    const std::string blob(RAW_BLOB);

    std::istringstream input_stream(blob);
    BlobSource source(input_stream);
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(source, true, nullptr, config));

    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    source = BlobSource(input_tensor);
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(source, true, nullptr, config));
}

/**
 * @brief If the magic bytes are present at the end of the input, then the factory can create "blob format v1" importers
 */
TEST_F(BlobFormatImportersTest, FactoryCanCreateImporterForBlobFormatV1) {
    const std::string blob = build_blob_format_v1_with_magic();
    std::istringstream input_stream(blob);
    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    BlobSource stream_source(input_stream);
    BlobSource tensor_source(input_tensor);

    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(stream_source, true, nullptr, config));
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(tensor_source, true, nullptr, config));

    stream_source.seekg(0, std::ios::beg);
    tensor_source.seekg(0, std::ios::beg);
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(stream_source, false, nullptr, config));
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(tensor_source, false, nullptr, config));

    stream_source.seekg(0, std::ios::beg);
    tensor_source.seekg(0, std::ios::beg);
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(stream_source, false, create_simple_model(), config));
    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(tensor_source, false, create_simple_model(), config));
}

/**
 * @brief A dummy model cannot be created before the graph
 */
TEST_F(BlobFormatImportersTest, CannotCreateModelBeforeGraph) {
    const std::string blob = build_blob_format_v1_with_magic();
    const ov::Tensor input_tensor(ov::element::Type_t::u8, ov::Shape({blob.size()}), blob.data());
    BlobSource source(input_tensor);

    OV_ASSERT_NO_THROW(importer = blob_format_importer_factory::create(source, true, nullptr, config));
    OV_EXPECT_THROW(importer->create_dummy_model(), ov::Exception, _);
}

/**
 * @brief A blob whose metadata declares the ELF format is always accepted
 */
TEST_F(BlobFormatImportersTest, FactoryAcceptsElfBlobType) {
    const std::string blob = build_blob_format_v1_with_blob_type(BlobType::ELF);
    std::istringstream input_stream(blob);
    BlobSource source(input_stream);

    OV_ASSERT_NO_THROW(blob_format_importer_factory::create(source, false, nullptr, config));
}

/**
 * @brief The blob type is read from unauthenticated metadata, and the host-executable formats are handed to the host
 * VM runtime instead of the NPU driver. Importing a blob that declares such a format must be refused unless the
 * application explicitly asked for it.
 */
TEST_F(BlobFormatImportersTest, FactoryRejectsHostExecutableBlobTypeByDefault) {
    for (const BlobType blob_type : {BlobType::LLVM, BlobType::BYTECODE}) {
        const std::string blob = build_blob_format_v1_with_blob_type(blob_type);
        std::istringstream input_stream(blob);
        BlobSource source(input_stream);

        OV_EXPECT_THROW(blob_format_importer_factory::create(source, false, nullptr, config), ov::Exception, _);
    }
}

/**
 * @brief A host-executable blob type is accepted once the application opted in
 */
TEST_F(BlobFormatImportersTest, FactoryAcceptsHostExecutableBlobTypeWhenAllowed) {
    config.update({{std::string(ALLOW_DYNAMIC_BLOB_IMPORT::key()), "YES"}});

    for (const BlobType blob_type : {BlobType::LLVM, BlobType::BYTECODE}) {
        const std::string blob = build_blob_format_v1_with_blob_type(blob_type);
        std::istringstream input_stream(blob);
        BlobSource source(input_stream);

        OV_ASSERT_NO_THROW(blob_format_importer_factory::create(source, false, nullptr, config));
    }
}
