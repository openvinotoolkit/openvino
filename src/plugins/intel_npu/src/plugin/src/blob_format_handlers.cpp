// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_handlers.hpp"

#include "intel_npu/common/parser_factory.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/runtime/allocator.hpp"

namespace {

ov::Tensor allocate_aligned_tensor(const size_t blobSize) {
    ov::Allocator customAllocator{intel_npu::utils::AlignedAllocator{intel_npu::utils::STANDARD_PAGE_SIZE}};
    ov::Tensor tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }

    return tensor;
}

/**
 * @brief Creates an "ov::Model" object which contains only the given "parameter" and "result" nodes.
 * @details Using an "ov::Model" object to create the "CompiledModel" is the preferred way of using the OV API.
 * This path allows making use of the already written functions/attributes for handling the I/O information.
 *
 * Note that a stored compiled model does not hold the original IR model within it. The only related information
 * which may be extracted is the original model's "parameter"/"result" nodes. Thus, we need to build a dummy model
 * starting from these fields in order to satisfy the API.
 *
 * @param inputDescriptors Describes the input nodes.
 * @param outputDescriptors Describes the output nodes.
 * @returns The dummy "ov::Model" composed of "parameter" and "result" nodes built using the given descriptors.
 */
std::shared_ptr<ov::Model> create_dummy_model(const std::vector<IODescriptor>& inputDescriptors,
                                              const std::vector<IODescriptor>& outputDescriptors,
                                              const std::optional<int64_t> batchSize,
                                              const std::optional<std::vector<ov::Layout>>& inputLayouts,
                                              const std::optional<std::vector<ov::Layout>>& outputLayouts) {
    ov::ParameterVector parameters;
    ov::ResultVector results;

    for (size_t inputIndex = 0; inputIndex < inputDescriptors.size(); ++inputIndex) {
        const IODescriptor& inputDescriptor = inputDescriptors.at(inputIndex);
        if (inputDescriptor.isStateInput || inputDescriptor.isStateOutput || inputDescriptor.isShapeTensor ||
            inputDescriptor.isInitInputWeights || inputDescriptor.isMainInputWeights) {
            continue;
        }

        auto shape = inputDescriptor.shapeFromIRModel.has_value() ? *inputDescriptor.shapeFromIRModel
                                                                  : inputDescriptor.shapeFromCompiler;

        if (batchSize.has_value()) {
            shape[intel_npu::utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
        }

        std::shared_ptr<ov::op::v0::Parameter> parameter =
            std::make_shared<ov::op::v0::Parameter>(inputDescriptor.precision, shape);

        parameter->set_friendly_name(inputDescriptor.nodeFriendlyName);
        parameter->output(0).get_tensor().set_names(inputDescriptor.outputTensorNames);
        if (inputLayouts.has_value()) {
            parameter->set_layout(inputLayouts->at(inputIndex));
        }
        parameters.push_back(std::move(parameter));
    }

    // The "result" nodes require a parent node in order to satisfy the API conventions. Additionally, a dummy shape for
    // the "Constant" node was required since the specific constructor does not accept "ov::PartialShape" values (a
    // constant can't have dynamic shape). The dummy tensor was also brought in order to register the correct,
    // potentially dynamic, output shape.
    for (size_t outputIndex = 0; outputIndex < outputDescriptors.size(); ++outputIndex) {
        const IODescriptor& outputDescriptor = outputDescriptors.at(outputIndex);
        if (outputDescriptor.isStateInput || outputDescriptor.isStateOutput || outputDescriptor.isShapeTensor ||
            outputDescriptor.isInitOutputWeights) {
            continue;
        }

        std::shared_ptr<ov::Node> constantDummy =
            std::make_shared<ov::op::v0::Constant>(outputDescriptor.precision, CONSTANT_NODE_DUMMY_SHAPE);

        auto shape = outputDescriptor.shapeFromIRModel.has_value() ? *outputDescriptor.shapeFromIRModel
                                                                   : outputDescriptor.shapeFromCompiler;

        if (batchSize.has_value()) {
            shape[intel_npu::utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
        }

        const std::shared_ptr<ov::descriptor::Tensor>& tensorDummy =
            std::make_shared<ov::descriptor::Tensor>(outputDescriptor.precision,
                                                     shape,
                                                     outputDescriptor.outputTensorNames);

        auto& result = results.emplace_back(std::make_shared<ov::op::v0::Result>(constantDummy));
        result->output(0).set_tensor_ptr(tensorDummy);
        if (outputLayouts.has_value()) {
            result->set_layout(outputLayouts->at(outputIndex));
        }
        result->set_friendly_name(outputDescriptor.nodeFriendlyName);
    }

    return std::make_shared<ov::Model>(results, parameters);
}

}  // namespace

namespace intel_npu {

IBlobFormatHandler::IBlobFormatHandler(const std::shared_ptr<ov::Model>& original_model,
                                       const FilteredConfig& config,
                                       const Logger& logger)
    : m_original_model(original_model),
      m_config(config),
      m_logger(logger) {}

std::shared_ptr<ov::Model> IBlobFormatHandler::create_dummy_model() const {
    OPENVINO_ASSERT(m_graph != nullptr, "Invalid state")
    return create_dummy_model(m_graph->get_metadata().inputs,
                              m_graph->get_metadata().outputs,
                              m_batch_size,
                              m_input_layouts,
                              m_output_layouts);
}

std::shared_ptr<IGraph> IBlobFormatHandler::create_graph(
    const std::shared_ptr<ZeroInitStructsHolder>& zero_init_structs) const {
    ParserFactory parserFactory;
    auto parser = parserFactory.getParser(zero_init_structs);

    const bool weights_separation_enabled = m_init_schedules.has_value();
    return parser->parse(m_main_schedule,
                         m_config,
                         m_init_schedules,
                         weights_separation_enabled ? m_original_model : std::nullopt,
                         get_compiler_compatibility_descriptor());
}

void IBlobFormatHandler::decrypt_schedules() {}

ov::Tensor IBlobFormatHandler::decrypt_schedule(const ov::Tensor& schedule) const {}

std::unordered_map<size_t, ov::Constant> IBlobFormatHandler::create_weights_map() const {}

RawBlobHandler::RawBlobHandler(std::istream& compiler_main_schedule,
                               const std::shared_ptr<ov::Model>& original_model,
                               const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("RawBlobHandler", config.get<LOG_LEVEL>())) {
    const size_t blob_size = MetadataBase::getFileSize(compiler_main_schedule);
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import is empty");

    m_main_schedule = allocate_aligned_tensor(blob_size);
    compiler_main_schedule.read(m_main_schedule.data<char>(), static_cast<std::streamsize>(blob_size));
}

RawBlobHandler::RawBlobHandler(const ov::Tensor& compiler_main_schedule,
                               const std::shared_ptr<ov::Model>& original_model,
                               const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("RawBlobHandler", config.get<LOG_LEVEL>())) {
    const size_t blob_size = compiler_main_schedule.get_byte_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import is empty");

    m_main_schedule = ov::Tensor(compiler_main_schedule, ov::Coordinate{0}, ov::Coordinate{blob_size});
}

ov::Tensor RawBlobHandler::extract_main_schedule() const {
    return m_main_schedule;
}

std::optional<std::vector<ov::Tensor>> RawBlobHandler::extract_init_schedules() const {
    return std::nullopt;
}

std::optional<int> RawBlobHandler::extract_batch_size() const {
    return std::nullopt;
}

std::optional<std::pair<std::vector<ov::Layout>>> RawBlobHandler::extract_layouts() const {
    return std::nullopt;
}

std::optional<std::string> RawBlobHandler::extract_compiler_compatibility_descriptor() const {
    return std::nullopt;
}

BlobFormatV1Handler::BlobFormatV1Handler(std::istream& npu_formatted_blob,
                                         const std::shared_ptr<ov::Model>& original_model,
                                         const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("BlobFormatV1Handler", config.get<LOG_LEVEL>())) {
    // Read only the metadata from the stream and check if the blob is compatible. Load the blob into memory only if
    // it passes the compatibility checks.
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import doesn't have any compiler payload");

    m_compiler_payload = allocate_aligned_tensor(blob_size);
    npu_formatted_blob.read(m_compiler_payload.data<char>(), static_cast<std::streamsize>(blob_size));
}

BlobFormatV1Handler::BlobFormatV1Handler(const ov::Tensor& npu_formatted_blob,
                                         const std::shared_ptr<ov::Model>& original_model,
                                         const FilteredConfig& config)
    : IBlobFormatHandler(original_model, config, Logger("BlobFormatV1Handler", config.get<LOG_LEVEL>())) {
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, "The blob provided for import doesn't have any compiler payload");

    // ROI tensor to skip the NPU plugin metadata
    m_compiler_payload = ov::Tensor(npu_formatted_blob, ov::Coordinate{0}, ov::Coordinate{blob_size});
}

ov::Tensor BlobFormatV1Handler::extract_main_schedule() const {
    const uint64_t main_size = metadata->get_main_schedule_size();

    return ov::Tensor(m_compiler_payload, ov::Coordinate{0}, ov::Coordinate{mainSize});
}

std::optional<std::vector<ov::Tensor>> BlobFormatV1Handler::extract_init_schedules() const {
    const std::optional<std::vector<uint64_t>> init_sizes = metadata->get_init_sizes();
    if (!init_sizes.has_value()) {
        return std::nullopt;
    }

    std::vector<ov::Tensor> init_schedules;
    size_t cursor_position = metadata->get_main_schedule_size();

    for (const uint64_t init_size : init_sizes.value()) {
        init_schedules.push_back(ov::Tensor(m_compiler_payload,
                                            ov::Coordinate{cursor_position},
                                            ov::Coordinate{cursor_position + init_size}));
        cursor_position += initSize;
    }

    return init_schedules;
}

std::optional<int> BlobFormatV1Handler::extract_batch_size() const {
    return metadata->get_batch_size();
}

std::optional<std::pair<std::vector<ov::Layout>>> BlobFormatV1Handler::extract_layouts() const {
    std::optional<std::vector<ov::Layout>> input_layouts = metadata->get_input_layouts();
    if (!input_layouts.has_value()) {
        return std::nullopt;
    }
    std::optional<std::vector<ov::Layout>> output_layouts = metadata->get_output_layouts();
    OPENVINO_ASSERT(output_layouts.has_value(),
                    "The metadata version received at import supports input layouts, but it doesn't support output "
                    "layouts. Either both or none should be supported")

    return std::make_pair<>(input_layouts.value(), output_layouts.value());
}

std::optional<std::string> BlobFormatV1Handler::extract_compiler_compatibility_descriptor() const {
    return metadata->get_compatibility_descriptor();
}

namespace blob_format_handler_factory {

std::shared_ptr<IBlobFormatHandler> create(std::istream& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config) {
    const Logger logger("blob_format_handler_factory", config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info("Blob compatibility check skipped.");

        return std::make_shared<RawBlobHandler>(npu_formatted_blob, original_model, config);
    }

    // The V1 format is identified by some magic bytes at the end of the input
    size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string blob_magic_bytes;
    blob_magic_bytes.resize(magic_bytes_size);

    std::streampos compiler_payload_beggining = npu_formatted_blob.tellg();
    npu_formatted_blob.seekg(-std::streampos(magic_bytes_size), std::ios::end);
    npu_formatted_blob.read(blob_magic_bytes.data(), magic_bytes_size);
    if (MAGIC_BYTES != blob_magic_bytes) {
        OPENVINO_THROW("The blob is missing the NPU metadata!");
    }

    npu_formatted_blob.seekg(compiler_payload_beggining, std::ios::beg);

    return std::make_shared<BlobFormatV1Handler>(npu_formatted_blob, original_model, config);
}

std::shared_ptr<IBlobFormatHandler> create(const ov::Tensor& npu_formatted_blob,
                                           const bool is_raw_blob,
                                           const std::shared_ptr<ov::Model>& original_model,
                                           const FilteredConfig& config) {
    const Logger logger("blob_format_handler_factory", config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info("Blob compatibility check skipped.");  // TODO string views

        return std::make_shared<RawBlobHandler>(npu_formatted_blob, original_model, config);
    }

    size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string_view blob_magic_bytes(
        npu_formatted_blob.data<const char>() + npu_formatted_blob.get_byte_size() - magic_bytes_size,
        magic_bytes_size);

    if (MAGIC_BYTES != blob_magic_bytes) {
        OPENVINO_THROW("The blob is missing the NPU metadata!");
    }

    return std::make_shared<BlobFormatV1Handler>(npu_formatted_blob, original_model, config);
}

}  // namespace blob_format_handler_factory

}  // namespace intel_npu
