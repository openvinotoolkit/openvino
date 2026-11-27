// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <string_view>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/utils/utils.hpp"
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

using blob_format_importer_utils::validate_main_schedule_descriptors;
using intel_npu::utils::DYNAMIC_BATCH_SIZE;

namespace {

IODescriptor create_descriptor(const std::string& name, const ov::PartialShape& shape) {
    IODescriptor descriptor;
    descriptor.nameFromCompiler = name;
    descriptor.precision = ov::element::f32;
    descriptor.shapeFromCompiler = shape;
    return descriptor;
}

}  // namespace
/**
 * @brief The init weights descriptors belong to the init schedules only, the main schedule must not contain any of
 * these regardless of their shapes or of the batch size
 */
TEST_F(BlobFormatImportersTest, MainScheduleRejectsInitWeightsDescriptors) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(create_descriptor("input", ov::Shape{1, 3}));
    metadata.outputs.push_back(create_descriptor("output", ov::Shape{1, 3}));
    IODescriptor main_weights = create_descriptor("main_weights", ov::Shape{});
    main_weights.isMainInputWeights = true;
    metadata.inputs.push_back(std::move(main_weights));
    OV_ASSERT_NO_THROW(validate_main_schedule_descriptors(metadata));

    IODescriptor init_input = create_descriptor("init_input", ov::Shape{1, 3});
    init_input.isInitInputWeights = true;
    metadata.inputs.push_back(std::move(init_input));
    OV_EXPECT_THROW_HAS_SUBSTRING(validate_main_schedule_descriptors(metadata), ov::Exception, "init_input");

    metadata.inputs.pop_back();
    IODescriptor init_output = create_descriptor("init_output", ov::Shape{1, 3});
    init_output.isInitOutputWeights = true;
    metadata.outputs.push_back(std::move(init_output));
    OV_EXPECT_THROW_HAS_SUBSTRING(validate_main_schedule_descriptors(metadata), ov::Exception, "init_output");
}

/**
 * @brief A static or dynamic batch size declared by the metadata means that the schedule was compiled with a batch of 1
 */
TEST_F(BlobFormatImportersTest, BatchSizeRequiresBatchOfOneInCompiledSchedule) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(create_descriptor("input", ov::Shape{1, 3}));
    metadata.outputs.push_back(create_descriptor("output", ov::Shape{1, 3}));

    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, 2));
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE));
    // no batch size declared by the metadata means that the batching is not handled by the plugin
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, std::nullopt));
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 0),
                                  ov::Exception,
                                  "invalid batch size");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE - 1),
                                  ov::Exception,
                                  "invalid batch size");

    // e.g. an output that does not depend on the batch, like the ShapeOf of a NonZero result
    metadata.outputs.push_back(create_descriptor("not_batched_output", ov::Shape{2}));
    // a dynamic batch is handled by the plugin in the same way as a static one, so the compiled schedule must have a
    // batch of 1 in this case as well, otherwise the outputs would be written out of bounds
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE),
                                  ov::Exception,
                                  "not_batched_output");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 2),
                                  ov::Exception,
                                  "not_batched_output");

    metadata.outputs.pop_back();
    metadata.inputs.push_back(create_descriptor("not_batched_input", ov::Shape{4, 3}));
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 4),
                                  ov::Exception,
                                  "not_batched_input");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE),
                                  ov::Exception,
                                  "not_batched_input");

    // the weights of the main schedule are not batched, thus they are skipped
    metadata.inputs.pop_back();
    IODescriptor main_weights = create_descriptor("main_weights", ov::Shape{5});
    main_weights.isMainInputWeights = true;
    metadata.inputs.push_back(std::move(main_weights));
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, 2));
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE));
    metadata.inputs.pop_back();

    // states and shape tensors cannot be combined with the plugin batching
    IODescriptor shape_tensor = create_descriptor("shape_tensor", ov::Shape{1});
    shape_tensor.isShapeTensor = true;
    metadata.inputs.push_back(std::move(shape_tensor));
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE),
                                  ov::Exception,
                                  "shape_tensor");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 2),
                                  ov::Exception,
                                  "shape_tensor");
    metadata.inputs.pop_back();

    IODescriptor state_input = create_descriptor("state_input", ov::Shape{1, 3});
    state_input.isStateInput = true;
    metadata.inputs.push_back(std::move(state_input));
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 2),
                                  ov::Exception,
                                  "state_input");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE),
                                  ov::Exception,
                                  "state_input");
    metadata.inputs.pop_back();

    IODescriptor state_output = create_descriptor("state_output", ov::Shape{1, 3});
    state_output.isStateOutput = true;
    metadata.outputs.push_back(std::move(state_output));
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, 2),
                                  ov::Exception,
                                  "state_output");
    OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE),
                                  ov::Exception,
                                  "state_output");
}

/**
 * @brief Dynamic dimensions other than the batch are accepted, as the batch is still 1 in the compiled schedule
 */
TEST_F(BlobFormatImportersTest, BatchSizeAcceptsDynamicDimensionsOtherThanBatch) {
    NetworkMetadata metadata;
    metadata.inputs.push_back(create_descriptor("input", ov::PartialShape{1, ov::Dimension(1, 10), -1}));
    metadata.outputs.push_back(create_descriptor("output", ov::PartialShape{1, ov::Dimension(1, 10)}));
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, 2));
    OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(metadata, DYNAMIC_BATCH_SIZE));
}

/**
 * @brief A declared batch requires every descriptor to have a static batch dimension of 1
 */
TEST_F(BlobFormatImportersTest, BatchSizeRejectsDescriptorsWithoutStaticBatchDimension) {
    const std::vector<std::pair<std::string, ov::PartialShape>> invalid_shapes = {
        {"dynamic_batch", ov::PartialShape{ov::Dimension(1, 10), 3}},
        {"unbounded_batch", ov::PartialShape{-1, 3}},
        {"dynamic_rank", ov::PartialShape::dynamic()},
        {"scalar", ov::PartialShape{}}};

    for (const auto& [name, shape] : invalid_shapes) {
        for (const int64_t batch_size : {int64_t{2}, DYNAMIC_BATCH_SIZE}) {
            NetworkMetadata input_metadata;
            input_metadata.inputs.push_back(create_descriptor(name, shape));
            OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(input_metadata, batch_size),
                                          ov::Exception,
                                          name);

            NetworkMetadata output_metadata;
            output_metadata.outputs.push_back(create_descriptor(name, shape));
            OV_EXPECT_THROW_HAS_SUBSTRING(blob_format_importer_utils::validate_batch_size(output_metadata, batch_size),
                                          ov::Exception,
                                          name);

            // nothing is checked if no batch size is declared
            OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(input_metadata, std::nullopt));
            OV_ASSERT_NO_THROW(blob_format_importer_utils::validate_batch_size(output_metadata, std::nullopt));
        }
    }
}
