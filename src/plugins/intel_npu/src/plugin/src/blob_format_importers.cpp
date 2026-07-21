// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/parser_factory.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/allocator.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view HANDLER_FACTOR_LOGGER_NAME = "blob_format_importer_factory";
constexpr std::string_view RAW_BLOB_HANDLER_LOGGER_NAME = "RawBlobImporter";
constexpr std::string_view BLOB_V1_HADNLER_LOGGER_NAME = "BlobFormatV1Importer";

constexpr std::string_view BLOB_COMPATIBILITY_SKIPPED_MESSAGE = "Blob compatibility check skipped.";
constexpr std::string_view MISSING_METADATA_MESSAGE = "The blob is missing the NPU metadata!";
constexpr std::string_view EMPTY_BLOB_MESSAGE = "The blob provided for import is empty";
constexpr std::string_view EMPTY_COMPILER_PAYLOAD_MESSAGE =
    "The blob provided for import doesn't have any compiler payload";
constexpr std::string_view DECRYPTING_PAYLOAD_MESSAGE = "Decrypting the compiler payload";

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

ov::Tensor allocate_aligned_tensor(size_t blobSize) {
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    ov::Tensor tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }

    return tensor;
}

/**
 * @brief Special case for PERF_COUNT as it requires compiler_type detection in case it is still set to PREFER_PLUGIN
 */
void update_compiler_type_if_perf_count(FilteredConfig& config,
                                        const ov::SoPtr<IEngineBackend>& backend,
                                        const std::string_view device_name) {
    if (config.has<PERF_COUNT>() && config.get<PERF_COUNT>() &&
        config.get<COMPILER_TYPE>() == ov::intel_npu::CompilerType::PREFER_PLUGIN) {
        ov::intel_npu::CompilerType compilerType = config.get<COMPILER_TYPE>();
        CompilerAdapterFactory factory;
        (void)factory.getCompiler(backend, compilerType, device_name);

        config.update({{ov::intel_npu::compiler_type.name(), COMPILER_TYPE::toString(compilerType)}});
    }
}

/**
 * @brief Uses the provided decryption callback to decrypt the given payload.
 */
void decrypt_payload(ov::Tensor& payload, const ov::EncryptionCallbacks& encryption_callbacks, const Logger& logger) {
    std::string decryptedBlobStr;
    {
        std::string encryptedBlobStr(payload.data<const char>(), payload.get_byte_size());  // +1x blob size
        decryptedBlobStr = encryption_callbacks.decrypt(encryptedBlobStr);                  // +1x blob size
    }  // -1x blob size when deallocating temporary encrypted blob string
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    size_t alignedSize = utils::align_size_to_standard_page_size(decryptedBlobStr.size());
    size_t paddingSize = alignedSize - decryptedBlobStr.size();
    payload = ov::Tensor(ov::element::u8, ov::Shape{alignedSize},
                         customAllocator);  // +1x blob size
    std::memcpy(payload.data<char>(), decryptedBlobStr.c_str(), decryptedBlobStr.size());
    if (paddingSize > 0) {
        // The blob obtained after decryption is expected to be the same as the blob we had before encryption.
        // That means blobs compiled with the current plugin version are expected to be already aligned.
        // However, the alignment might not be mandatory in a future plugin version. For this scenario, the
        // padding is added here in order to make use of this "non-copy optimization".
        logger.warning("Decrypted blob size was not page aligned, additional %zu bytes padding will be added",
                       paddingSize);
        std::memset(payload.data<char>() + decryptedBlobStr.size(), 0, paddingSize);
    }
}  // -1x blob size when deallocating decrypted blob string

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
                                              const std::optional<int> batchSize,
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
            shape[utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
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
            shape[utils::BATCH_AXIS] = ov::Dimension(batchSize.value());
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

IBlobFormatImporter::IBlobFormatImporter(const std::shared_ptr<const ov::Model>& original_model,
                                         const FilteredConfig& config,
                                         const Logger& logger)
    : m_config(config),
      m_logger(logger),
      m_original_model(original_model) {}

std::shared_ptr<IGraph> IBlobFormatImporter::create_graph(const ov::SoPtr<IEngineBackend>& backend,
                                                          const std::string_view network_name,
                                                          const std::string_view device_name,
                                                          const std::shared_ptr<ov::ICore>& core) {
    OV_ITT_TASK_CHAIN(PARSE_AND_CREATE_GRAPH, itt::domains::NPUPlugin, "IBlobFormatImporter", "create_graph");
    m_logger.debug("Creating a graph");

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "decrypt_schedules");
    decrypt_schedules();

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "extract_main_schedule");
    const ov::Tensor main_schedule = extract_main_schedule();

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "extract_init_schedules");
    const std::optional<std::vector<ov::Tensor>> init_schedules = extract_init_schedules();
    m_batch_size = extract_batch_size();

    update_compiler_type_if_perf_count(m_config, backend, device_name);

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "get_parser");
    m_logger.trace("Creating the parser");
    ParserFactory parserFactory;
    auto parser = parserFactory.getParser(backend->getInitStructs());

    std::variant<std::monostate, std::shared_ptr<const ov::Model>, std::pair<std::string, std::shared_ptr<ov::ICore>>>
        weights_source;
    if (init_schedules.has_value()) {
        if (m_original_model) {
            weights_source = std::move(m_original_model);
        } else if (!m_config.get<WEIGHTS_PATH>().empty()) {
            weights_source = std::make_pair<>(m_config.get<WEIGHTS_PATH>(), core);
        } else {
            OPENVINO_THROW("Attempted to load a weightless compiled model, but no weights have been provided");
        }
    }

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "parse");
    m_logger.trace("Calling the parser");
    m_graph = parser->parse(main_schedule,
                            m_config,
                            std::move(weights_source),
                            init_schedules,
                            extract_compiler_compatibility_descriptor());

    m_graph->update_network_name(network_name);
    if (m_batch_size.has_value() && m_batch_size.value() > 0) {
        // Initial batch setup for static cases
        m_graph->set_batch_size(m_batch_size.value());
    }

    return m_graph;
}

std::shared_ptr<ov::Model> IBlobFormatImporter::create_dummy_model() const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "IBlobFormatImporter::create_dummy_model");
    m_logger.debug("Creating a dummy ov::Model");

    OPENVINO_ASSERT(m_graph != nullptr, "Invalid state");

    const std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> layouts = extract_layouts();
    return ::create_dummy_model(m_graph->get_metadata().inputs,
                                m_graph->get_metadata().outputs,
                                m_batch_size,
                                layouts.has_value() ? std::make_optional<>(layouts->first) : std::nullopt,
                                layouts.has_value() ? std::make_optional<>(layouts->second) : std::nullopt);
}

FilteredConfig IBlobFormatImporter::get_config() const {
    return m_config;
}

void IBlobFormatImporter::log_contents(const std::optional<std::string>& compatibility_descriptor) {
    // TODO?
}

RawBlobImporter::RawBlobImporter(std::istream& compiler_main_schedule,
                                 const std::shared_ptr<const ov::Model>& original_model,
                                 const FilteredConfig& config)
    : IBlobFormatImporter(original_model,
                          config,
                          Logger(RAW_BLOB_HANDLER_LOGGER_NAME.data(), config.get<LOG_LEVEL>())) {
    const size_t blob_size = MetadataBase::getFileSize(compiler_main_schedule);
    OPENVINO_ASSERT(blob_size > 0, EMPTY_BLOB_MESSAGE);

    m_main_schedule = allocate_aligned_tensor(blob_size);
    compiler_main_schedule.read(m_main_schedule.data<char>(), static_cast<std::streamsize>(blob_size));
}

RawBlobImporter::RawBlobImporter(const ov::Tensor& compiler_main_schedule,
                                 const std::shared_ptr<const ov::Model>& original_model,
                                 const FilteredConfig& config)
    : IBlobFormatImporter(original_model,
                          config,
                          Logger(RAW_BLOB_HANDLER_LOGGER_NAME.data(), config.get<LOG_LEVEL>())) {
    const size_t blob_size = compiler_main_schedule.get_byte_size();
    OPENVINO_ASSERT(blob_size > 0, EMPTY_BLOB_MESSAGE);

    m_main_schedule = ov::Tensor(compiler_main_schedule, ov::Coordinate{0}, ov::Coordinate{blob_size});
}

void RawBlobImporter::decrypt_schedules() {
    const bool is_null_decryption = !(m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                      m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt != nullptr);
    if (is_null_decryption) {
        m_logger.debug("No decryption callback found");
        return;
    }

    m_logger.warning("Received decryption callback, but metadata parsing is skipped and cannot determine if blob was "
                     "encrypted or not.");

    m_logger.debug(DECRYPTING_PAYLOAD_MESSAGE.data());
    decrypt_payload(m_main_schedule, m_config.get<CACHE_ENCRYPTION_CALLBACKS>(), m_logger);
}

ov::Tensor RawBlobImporter::extract_main_schedule() const {
    return m_main_schedule;
}

std::optional<std::vector<ov::Tensor>> RawBlobImporter::extract_init_schedules() const {
    return std::nullopt;
}

std::optional<int> RawBlobImporter::extract_batch_size() const {
    return std::nullopt;
}

std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> RawBlobImporter::extract_layouts() const {
    return std::nullopt;
}

std::optional<std::string> RawBlobImporter::extract_compiler_compatibility_descriptor() const {
    return std::nullopt;
}

BlobFormatV1Importer::BlobFormatV1Importer(std::istream& npu_formatted_blob,
                                           const std::shared_ptr<const ov::Model>& original_model,
                                           const FilteredConfig& config)
    : IBlobFormatImporter(original_model, config, Logger(BLOB_V1_HADNLER_LOGGER_NAME.data(), config.get<LOG_LEVEL>())) {
    // Read only the metadata from the stream and check if the blob is compatible. Load the blob into memory only if
    // it passes the compatibility checks.
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, EMPTY_COMPILER_PAYLOAD_MESSAGE);

    m_compiler_payload = allocate_aligned_tensor(blob_size);
    npu_formatted_blob.read(m_compiler_payload.data<char>(), static_cast<std::streamsize>(blob_size));

    register_compiler_version();
}

BlobFormatV1Importer::BlobFormatV1Importer(const ov::Tensor& npu_formatted_blob,
                                           const std::shared_ptr<const ov::Model>& original_model,
                                           const FilteredConfig& config)
    : IBlobFormatImporter(original_model, config, Logger(BLOB_V1_HADNLER_LOGGER_NAME.data(), config.get<LOG_LEVEL>())) {
    m_metadata = read_metadata_from(npu_formatted_blob);

    const size_t blob_size = m_metadata->get_blob_size();
    OPENVINO_ASSERT(blob_size > 0, EMPTY_COMPILER_PAYLOAD_MESSAGE);

    // ROI tensor to skip the NPU plugin metadata
    m_compiler_payload = ov::Tensor(npu_formatted_blob, ov::Coordinate{0}, ov::Coordinate{blob_size});

    register_compiler_version();
}

void BlobFormatV1Importer::decrypt_schedules() {
    const bool is_payload_encrypted = m_metadata->is_encrypted_blob().value_or(false);
    const bool is_null_decryption = !(m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                      m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt != nullptr);
    if (!is_payload_encrypted) {
        m_logger.debug("The compiler payload is NOT encrypted");
        return;
    }
    OPENVINO_ASSERT(!is_null_decryption, "Blob is encrypted, but no decryption callback was provided!");

    m_logger.debug(DECRYPTING_PAYLOAD_MESSAGE.data());
    decrypt_payload(m_compiler_payload, m_config.get<CACHE_ENCRYPTION_CALLBACKS>(), m_logger);
}

ov::Tensor BlobFormatV1Importer::extract_main_schedule() const {
    const uint64_t main_size = m_metadata->get_main_schedule_size();

    return ov::Tensor(m_compiler_payload, ov::Coordinate{0}, ov::Coordinate{main_size});
}

std::optional<std::vector<ov::Tensor>> BlobFormatV1Importer::extract_init_schedules() const {
    const std::optional<std::vector<uint64_t>> init_sizes = m_metadata->get_init_sizes();
    if (!init_sizes.has_value()) {
        return std::nullopt;
    }

    std::vector<ov::Tensor> init_schedules;
    size_t cursor_position = m_metadata->get_main_schedule_size();

    m_logger.debug("Extracting %zu init schedules", init_sizes->size());

    for (const uint64_t init_size : init_sizes.value()) {
        m_logger.debug("Init size: %llu", init_size);

        init_schedules.push_back(ov::Tensor(m_compiler_payload,
                                            ov::Coordinate{cursor_position},
                                            ov::Coordinate{cursor_position + init_size}));
        cursor_position += init_size;
    }

    return init_schedules;
}

std::optional<int> BlobFormatV1Importer::extract_batch_size() const {
    const std::optional<int64_t> batch_size = m_metadata->get_batch_size();
    if (batch_size.has_value()) {
        m_logger.debug("Extracted batch size: %d", batch_size.value());
        return std::make_optional<int>(static_cast<int>(batch_size.value()));
    }
    return std::nullopt;
}

std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> BlobFormatV1Importer::extract_layouts()
    const {
    std::optional<std::vector<ov::Layout>> input_layouts = m_metadata->get_input_layouts();
    if (!input_layouts.has_value()) {
        return std::nullopt;
    }
    std::optional<std::vector<ov::Layout>> output_layouts = m_metadata->get_output_layouts();
    OPENVINO_ASSERT(output_layouts.has_value(),
                    "The metadata version received at import supports input layouts, but it doesn't support output "
                    "layouts. Either both or none should be supported");

    return std::make_pair<>(input_layouts.value(), output_layouts.value());
}

std::optional<std::string> BlobFormatV1Importer::extract_compiler_compatibility_descriptor() const {
    const std::optional<std::string_view> compatibility_descriptor = m_metadata->get_compatibility_descriptor();
    // Convert the descriptor to an owning string before the metadata is potentially destroyed
    return compatibility_descriptor.has_value() ? std::make_optional<>(std::string(compatibility_descriptor.value()))
                                                : std::nullopt;
}

void BlobFormatV1Importer::register_compiler_version() {
    std::optional<uint32_t> compiler_version = m_metadata->get_compiler_version();
    if (compiler_version.has_value()) {
        m_config.update({{ov::intel_npu::compiler_version.name(), std::to_string(compiler_version.value())}});
        m_logger.debug("Imported model was compiled with compiler version: %u.%u",
                       ONEAPI_VERSION_MAJOR(compiler_version.value()),
                       ONEAPI_VERSION_MINOR(compiler_version.value()));
    }
}

namespace blob_format_importer_factory {

std::unique_ptr<IBlobFormatImporter> create(std::istream& npu_formatted_blob,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const FilteredConfig& config) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "blob_format_importer_factory::create(std::istream)");

    const Logger logger(HANDLER_FACTOR_LOGGER_NAME.data(), config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info(BLOB_COMPATIBILITY_SKIPPED_MESSAGE.data());

        logger.debug("Creating a raw blob format import handler from a stream using the factory");
        return std::make_unique<RawBlobImporter>(npu_formatted_blob, original_model, config);
    }

    // The V1 format is identified by some magic bytes at the end of the input
    const size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string blob_magic_bytes;
    blob_magic_bytes.resize(magic_bytes_size);

    std::streampos compiler_payload_beggining = npu_formatted_blob.tellg();
    npu_formatted_blob.seekg(-std::streampos(magic_bytes_size), std::ios::end);
    npu_formatted_blob.read(blob_magic_bytes.data(), magic_bytes_size);

    OPENVINO_ASSERT(MAGIC_BYTES == blob_magic_bytes, MISSING_METADATA_MESSAGE);

    npu_formatted_blob.seekg(compiler_payload_beggining, std::ios::beg);

    logger.debug("Creating a blob format v1 import handler from a stream using the factory");
    return std::make_unique<BlobFormatV1Importer>(npu_formatted_blob, original_model, config);
}

std::unique_ptr<IBlobFormatImporter> create(const ov::Tensor& npu_formatted_blob,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const FilteredConfig& config) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "blob_format_importer_factory::create(ov::Tensor)");

    const Logger logger(HANDLER_FACTOR_LOGGER_NAME.data(), config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info(BLOB_COMPATIBILITY_SKIPPED_MESSAGE.data());

        logger.debug("Creating a raw blob format import handler from a tensor using the factory");
        return std::make_unique<RawBlobImporter>(npu_formatted_blob, original_model, config);
    }

    size_t magic_bytes_size = MAGIC_BYTES.size();
    std::string_view blob_magic_bytes(
        npu_formatted_blob.data<const char>() + npu_formatted_blob.get_byte_size() - magic_bytes_size,
        magic_bytes_size);

    OPENVINO_ASSERT(MAGIC_BYTES == blob_magic_bytes, MISSING_METADATA_MESSAGE);

    logger.debug("Creating a blob format v1 import handler from a tensor using the factory");
    return std::make_unique<BlobFormatV1Importer>(npu_formatted_blob, original_model, config);
}

}  // namespace blob_format_importer_factory

}  // namespace intel_npu
