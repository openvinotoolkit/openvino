// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
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

std::shared_ptr<ov::Model> create_simple_model() {
    auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{5}, std::vector<float>{1.0f});
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto add = std::make_shared<ov::op::v1::Add>(input, weights);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input}, "Simple with weights");
}

}  // namespace

using testing::_;

struct BlobFormatImportersTest : public ::testing::Test {
    BlobFormatImportersTest() : config(std::make_shared<OptionsDesc>()) {}

    std::unique_ptr<IBlobFormatImporter> importer;
    Config config;
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

namespace {

IODescriptor create_descriptor(const std::string& name, const ov::PartialShape& shape_from_compiler) {
    IODescriptor descriptor;
    descriptor.nameFromCompiler = name;
    descriptor.precision = ov::element::f32;
    descriptor.shapeFromCompiler = shape_from_compiler;
    return descriptor;
}

/**
 * @brief Network metadata as reported by the driver for a schedule compiled with batch handled by the plugin, i.e. the
 * compiled schedule processes one batch slice at a time.
 */
NetworkMetadata create_plugin_batched_metadata() {
    NetworkMetadata metadata;
    metadata.inputs.push_back(create_descriptor("input", ov::PartialShape{1, 3, 8, 8}));
    metadata.outputs.push_back(create_descriptor("output", ov::PartialShape{1, 10}));
    return metadata;
}

}  // namespace

using blob_format_importer_utils::DYNAMIC_BATCH_SIZE;
using blob_format_importer_utils::validate_batch_size;

/**
 * @brief Static and dynamic batch sizes are accepted if the schedule was compiled with a batch of 1
 */
TEST_F(BlobFormatImportersTest, BatchSizeAcceptedForPluginBatchedSchedule) {
    const NetworkMetadata metadata = create_plugin_batched_metadata();

    for (const int64_t batch_size : {int64_t{1}, int64_t{3}, int64_t{4294967297}, DYNAMIC_BATCH_SIZE}) {
        EXPECT_NO_THROW(validate_batch_size(batch_size, metadata)) << "batch size: " << batch_size;
    }
}

/**
 * @brief Values outside of the {-1} U [1, +inf) domain are rejected
 */
TEST_F(BlobFormatImportersTest, BatchSizeInvalidValuesRejected) {
    const NetworkMetadata metadata = create_plugin_batched_metadata();

    for (const int64_t batch_size : {int64_t{0}, int64_t{-2}, std::numeric_limits<int64_t>::min()}) {
        SCOPED_TRACE("batch size: " + std::to_string(batch_size));
        OV_EXPECT_THROW(validate_batch_size(batch_size, metadata), ov::Exception, _);
    }
}

/**
 * @brief A schedule compiled with a batch different than 1 must not be combined with a batch size from the metadata.
 * Otherwise, the buffers are sized according to the metadata, while the device reads and writes them according to the
 * compiled schedule.
 */
TEST_F(BlobFormatImportersTest, BatchSizeRejectedForCompilerBatchedSchedule) {
    NetworkMetadata input_batched = create_plugin_batched_metadata();
    input_batched.inputs.at(0).shapeFromCompiler = ov::PartialShape{3, 3, 8, 8};

    NetworkMetadata output_batched = create_plugin_batched_metadata();
    output_batched.outputs.at(0).shapeFromCompiler = ov::PartialShape{3, 10};

    NetworkMetadata dynamic_batch = create_plugin_batched_metadata();
    dynamic_batch.inputs.at(0).shapeFromCompiler = ov::PartialShape{ov::Dimension::dynamic(), 3, 8, 8};

    NetworkMetadata scalar = create_plugin_batched_metadata();
    scalar.outputs.at(0).shapeFromCompiler = ov::PartialShape{};

    for (const NetworkMetadata& metadata : {input_batched, output_batched, dynamic_batch, scalar}) {
        for (const int64_t batch_size : {int64_t{1}, int64_t{3}, int64_t{4294967297}, DYNAMIC_BATCH_SIZE}) {
            SCOPED_TRACE("batch size: " + std::to_string(batch_size));
            OV_EXPECT_THROW(validate_batch_size(batch_size, metadata), ov::Exception, _);
        }
    }
}

/**
 * @brief States and shape tensors are not sliced per batch, thus they cannot coexist with a batch size from the
 * metadata
 */
TEST_F(BlobFormatImportersTest, BatchSizeRejectedForStatesAndShapeTensors) {
    NetworkMetadata state_input = create_plugin_batched_metadata();
    state_input.inputs.push_back(create_descriptor("state", ov::PartialShape{1, 16}));
    state_input.inputs.back().isStateInput = true;

    NetworkMetadata state_output = create_plugin_batched_metadata();
    state_output.outputs.push_back(create_descriptor("state", ov::PartialShape{1, 16}));
    state_output.outputs.back().isStateOutput = true;

    NetworkMetadata shape_tensor = create_plugin_batched_metadata();
    shape_tensor.inputs.push_back(create_descriptor("input", ov::PartialShape{1}));
    shape_tensor.inputs.back().isShapeTensor = true;

    for (const NetworkMetadata& metadata : {state_input, state_output, shape_tensor}) {
        OV_EXPECT_THROW(validate_batch_size(1, metadata), ov::Exception, _);
    }
}

/**
 * @brief The compilation path stores -1 as batch size for models with a dynamic batch. The metadata must preserve this
 * value (only 0 is mapped to "no batch size").
 */
TEST_F(BlobFormatImportersTest, DynamicBatchSizeSurvivesMetadataRoundTrip) {
    std::ostringstream stream;
    stream << DUMMY_COMPILER_PAYLOAD.data();
    Metadata<CURRENT_METADATA_VERSION>(DUMMY_COMPILER_PAYLOAD.size(),
                                       CURRENT_OPENVINO_VERSION,
                                       std::nullopt,
                                       DYNAMIC_BATCH_SIZE)
        .write(stream);
    const std::string blob = stream.str();

    std::istringstream input_stream(blob);
    BlobSource source(input_stream);
    const std::unique_ptr<MetadataBase> metadata = read_metadata_from(source);
    ASSERT_NE(metadata, nullptr);
    EXPECT_EQ(metadata->get_batch_size(), std::make_optional<int64_t>(DYNAMIC_BATCH_SIZE));
}
