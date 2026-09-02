// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include "batch_size_section.hpp"
#include "compiler_option_support_helper.hpp"
#include "compiler_schedule_instance_evaluator.hpp"
#include "compiler_schedules_sections.hpp"
#include "compiler_version_section.hpp"
#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "intel_npu/common/encrypted_schedules_flag_section.hpp"
#include "intel_npu/common/isection.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/parser_factory.hpp"
#include "intel_npu/common/supported_section_type_evaluator.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/utils.hpp"
#include "io_layouts_section.hpp"
#include "metadata.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/allocator.hpp"

namespace {

using namespace intel_npu;

// TODO make utility
ov::Tensor allocate_aligned_tensor(size_t blobSize) {
    ov::Allocator customAllocator{utils::AlignedAllocator{utils::STANDARD_PAGE_SIZE}};
    if (blobSize > static_cast<decltype(blobSize)>(std::numeric_limits<std::streamsize>::max())) {
        OPENVINO_THROW("Blob size is too large to be represented on a std::streamsize!");
    }

    return ov::Tensor(ov::element::u8, ov::Shape{blobSize}, customAllocator);
}

constexpr std::string_view BLOB_COMPATIBILITY_SKIPPED_MESSAGE = "Blob compatibility check skipped.";
constexpr std::string_view EMPTY_BLOB_MESSAGE = "The blob provided for import is empty";
constexpr std::string_view BLOB_SIZE_SMALLER_THAN_MAGIC =
    "Received a blob for import that is not a raw one and its size is smaller than the size of the magic bytes";
constexpr std::string_view EMPTY_COMPILER_PAYLOAD_MESSAGE =
    "The blob provided for import doesn't have any compiler payload";
constexpr std::string_view DECRYPTING_PAYLOAD_MESSAGE = "Decrypting the compiler payload";
constexpr std::string_view NEW_PAGE_ALIGNED_BUFFER_MESSAGE =
    "The compiler payload has been copied into a new, page aligned buffer";
constexpr std::string_view MISSING_MAIN_SCHEDULE_MESSAGE = "The compiler main schedule is missing";
constexpr std::string_view GRAPH_CLASS_MISMATCH_MESSAGE = "The blob type doesn't match the type of \"Graph\"";

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

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

// TODO make utility, also used within compiler schedules
/**
 * @brief Uses the provided decryption callback to decrypt the given payload.
 */
void decrypt_payload(ov::Tensor& payload, const ov::EncryptionCallbacks& encryption_callbacks, const Logger& logger) {
    OPENVINO_ASSERT(encryption_callbacks.decrypt, "Decryption requested without providing a decryption callback");

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

// TODO move importers in separate files, another new directory. do not expose them tho

/**
 * @brief Class used to import a blob that contains only the compiler main schedule.
 */
class RawBlobImporter : public IBlobFormatImporter {
public:
    explicit RawBlobImporter(BlobSource& compiler_main_schedule,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const ov::SoPtr<IEngineBackend>& backend,
                             const FilteredConfig& config)
        : IBlobFormatImporter(original_model, backend, config, Logger("RawBlobImporter", config.get<LOG_LEVEL>())) {
        const size_t blob_size = compiler_main_schedule.get_remaining_size();
        OPENVINO_ASSERT(blob_size > 0, EMPTY_BLOB_MESSAGE);

        if (!compiler_main_schedule.is_contiguous()) {
            m_main_schedule = allocate_aligned_tensor(blob_size);
            compiler_main_schedule.read_into_buffer(m_main_schedule.data(), blob_size);

            m_logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data());
            return;
        }

        m_main_schedule = compiler_main_schedule.create_roi_tensor(blob_size);
    }

    std::shared_ptr<BlobWriter> create_blob_writer() override {
        return nullptr;
    }

private:
    /**
     * @brief Decrypts the compiler main schedule if a decryption callback was received.
     */
    void decrypt_schedules() override {
        const bool is_null_decryption = !(m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                          m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt != nullptr);
        if (is_null_decryption) {
            m_logger.debug("No decryption callback found");
            return;
        }

        m_logger.warning(
            "Received decryption callback, but metadata parsing is skipped and cannot determine if blob was "
            "encrypted or not.");

        m_logger.debug(DECRYPTING_PAYLOAD_MESSAGE.data());
        decrypt_payload(m_main_schedule, m_config.get<CACHE_ENCRYPTION_CALLBACKS>(), m_logger);
    }

    ov::Tensor extract_main_schedule() const override {
        return m_main_schedule;
    }

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override {
        return std::nullopt;
    }

    std::optional<int> extract_batch_size() const override {
        return std::nullopt;
    }

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override {
        return std::nullopt;
    }

    std::optional<uint32_t> extract_compiler_version() const override {
        return std::nullopt;
    }

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override {
        return std::nullopt;
    }

    std::optional<BlobType> extract_blob_type() const override {
        return std::nullopt;
    }

    /**
     * @brief The compiler main schedule, that is also the whole blob received to be imported.
     */
    ov::Tensor m_main_schedule;
};

/**
 * @brief Class used to import a blob that follows the "V1" format: compiler payload + some (non-TLV) metadata
 */
class BlobFormatV1Importer : public IBlobFormatImporter {
public:
    explicit BlobFormatV1Importer(BlobSource& npu_formatted_blob,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const ov::SoPtr<IEngineBackend>& backend,
                                  const FilteredConfig& config)
        : IBlobFormatImporter(original_model,
                              backend,
                              config,
                              Logger("BlobFormatV1Importer", config.get<LOG_LEVEL>())) {
        // Read only the metadata from the source and check if the blob is compatible. Load the blob into memory only if
        // it passes the compatibility checks.
        m_metadata = read_metadata_from(npu_formatted_blob);

        const size_t compiler_payload_size = m_metadata->get_compiler_payload_size();
        OPENVINO_ASSERT(compiler_payload_size > 0, EMPTY_COMPILER_PAYLOAD_MESSAGE);

        if (!npu_formatted_blob.is_contiguous()) {
            m_compiler_payload = allocate_aligned_tensor(compiler_payload_size);
            npu_formatted_blob.read_into_buffer(m_compiler_payload.data(), compiler_payload_size);

            m_logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data());
        } else {
            // ROI tensor to skip the NPU plugin metadata
            m_compiler_payload = npu_formatted_blob.create_roi_tensor(compiler_payload_size);
        }
    }

    // TODO tests for this
    /**
     * @brief Constructs a blob writer that can be used to re-export the blob.
     * @details The sections are built using the content found within the V1 format.
     */
    std::shared_ptr<BlobWriter> create_blob_writer() override {
        OPENVINO_ASSERT(m_graph, "Invalid state");

        auto blob_writer = std::make_shared<BlobWriter>();

        // Register the compiler schedules
        const bool encryption_enabled = m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                        m_config.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr;
        const std::optional<ov::EncryptionCallbacks> encryption_callbacks =
            encryption_enabled ? std::make_optional<>(m_config.get<CACHE_ENCRYPTION_CALLBACKS>()) : std::nullopt;

        switch (m_graph->get_kind()) {
        case GraphKind::Dynamic: {
            auto dynamic_graph = std::dynamic_pointer_cast<DynamicGraph>(m_graph);
            OPENVINO_ASSERT(dynamic_graph, GRAPH_CLASS_MISMATCH_MESSAGE);
            blob_writer->register_section(
                std::make_shared<DynamicScheduleSection>(dynamic_graph, encryption_callbacks, m_logger.level()));
            break;
        }
        case GraphKind::Weightless: {
            auto weightless_graph = std::dynamic_pointer_cast<WeightlessGraph>(m_graph);
            OPENVINO_ASSERT(weightless_graph, GRAPH_CLASS_MISMATCH_MESSAGE);
            blob_writer->register_section(
                std::make_shared<ELFInitSchedulesSection>(weightless_graph, encryption_callbacks, m_logger.level()));
        }
        case GraphKind::Weightful: {
            auto graph = std::dynamic_pointer_cast<Graph>(m_graph);
            OPENVINO_ASSERT(graph, GRAPH_CLASS_MISMATCH_MESSAGE);
            blob_writer->register_section(
                std::make_shared<ELFMainScheduleSection>(graph, encryption_callbacks, m_logger.level()));
            break;
        }
        default: {
            OPENVINO_THROW("Unsupported kind of \"Graph\"");
        }
        }

        // Miscellaneous
        if (m_batch_size.has_value()) {
            blob_writer->register_section(std::make_shared<BatchSizeSection>(m_batch_size.value(), m_logger.level()));
        }

        const auto layouts = extract_layouts();
        if (layouts.has_value()) {
            blob_writer->register_section(
                std::make_shared<IOLayoutsSection>(layouts->first, layouts->second, m_logger.level()));
        }

        const auto compiler_version = extract_compiler_version();
        if (compiler_version.has_value()) {
            blob_writer->register_section(
                std::make_shared<CompilerVersionSection>(compiler_version.value(), m_logger.level()));
        }

        if (encryption_enabled) {
            blob_writer->register_section(
                std::make_shared<EncryptedSchedulesFlagSection>(encryption_enabled, m_logger.level()));
        }

        // TODO compatibility reqs section

        return blob_writer;
    }

private:
    /**
     * @brief Decrypts the whole compiler payload (main schedule + init schedules if applicable) if:
     *   1. A decryption callback was provided and
     *   2. The metadata indicates the blob was encrypted.
     * @throws ov::AssertFailure if the blob was encrypted but no decryption callback was provided.
     */
    void decrypt_schedules() override {
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

    // TODO check blob ownership management
    ov::Tensor extract_main_schedule() const override {
        const uint64_t main_size = m_metadata->get_main_schedule_size();

        return ov::Tensor(m_compiler_payload, ov::Coordinate{0}, ov::Coordinate{main_size});
    }

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override {
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

    std::optional<int> extract_batch_size() const override {
        const std::optional<int64_t> batch_size = m_metadata->get_batch_size();
        if (batch_size.has_value()) {
            m_logger.debug("Extracted batch size: %d", batch_size.value());
            return std::make_optional<int>(static_cast<int>(batch_size.value()));
        }
        return std::nullopt;
    }

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override {
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

    std::optional<uint32_t> extract_compiler_version() const override {
        return m_metadata->get_compiler_version();
    }

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override {
        const std::optional<std::string_view> compatibility_descriptor = m_metadata->get_compatibility_descriptor();
        // Convert the descriptor to an owning string before the metadata is potentially destroyed
        return compatibility_descriptor.has_value()
                   ? std::make_optional<>(std::string(compatibility_descriptor.value()))
                   : std::nullopt;
    }

    std::optional<BlobType> extract_blob_type() const override {
        return m_metadata->get_blob_type();
    }

    /**
     * @brief The whole compiler payload. Init schedules include if weights separation was used.
     */
    ov::Tensor m_compiler_payload;
    std::unique_ptr<MetadataBase> m_metadata;
};

/**
 * @brief Class used to import a blob that follows the "V2" format: header + sections + manifest (HSM)
 */
class BlobFormatV2Importer : public IBlobFormatImporter {
public:
    explicit BlobFormatV2Importer(BlobSource& npu_formatted_blob,
                                  const std::shared_ptr<const ov::Model>& original_model,
                                  const ov::SoPtr<IEngineBackend>& backend,
                                  const std::shared_ptr<CompilerOptionSupportHelper>& option_helper,
                                  const FilteredConfig& config)
        : IBlobFormatImporter(original_model, backend, config, Logger("BlobFormatV2Importer", config.get<LOG_LEVEL>())),
          m_blob_reader(config) {
        register_known_sections_and_evaluators(option_helper);

        m_blob_reader.read(npu_formatted_blob);
        verify_valid_sections();
    }

    std::shared_ptr<BlobWriter> create_blob_writer() override {
        // Moving forward, the "Graph" object will manage the ownership of the compiler schedules
        auto elfMainScheduleSection = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE));

        if (elfMainScheduleSection) {
            auto initSchedulesSection = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
                m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));

            elfMainScheduleSection->set_graph(std::dynamic_pointer_cast<Graph>(m_graph));
            if (initSchedulesSection) {
                initSchedulesSection->set_graph(std::dynamic_pointer_cast<WeightlessGraph>(m_graph));
            }
        } else {
            auto dynamicScheduleSection = std::dynamic_pointer_cast<DynamicScheduleSection>(
                m_blob_reader.retrieve_first_section(PredefinedSectionType::DYNAMIC_SCHEDULE));
            dynamicScheduleSection->set_graph(std::dynamic_pointer_cast<DynamicGraph>(m_graph));
        }

        return std::make_shared<BlobWriter>(m_blob_reader);
    }

private:
    /**
     * @brief Registers all blob sections readers known to the plugin.
     * @note The CRE & Manifest sections should have been already registered (e.g. in the BlobReader ctor) since
     * these sections are a core part of the format.
     */
    void register_known_sections_and_evaluators(const std::shared_ptr<CompilerOptionSupportHelper>& option_helper) {
        // TODO shotgun surgery? should these correspond to the "supported" section types?
        m_blob_reader.register_reader(PredefinedSectionType::ELF_MAIN_SCHEDULE, ELFMainScheduleSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::ELF_INIT_SCHEDULES, ELFInitSchedulesSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::DYNAMIC_SCHEDULE, ELFInitSchedulesSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::BATCH_SIZE, BatchSizeSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::IO_LAYOUTS, IOLayoutsSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::ENCRYPTED_SCHEDULES_FLAG, IOLayoutsSection::read);
        m_blob_reader.register_reader(PredefinedSectionType::COMPILER_VERSION, IOLayoutsSection::read);

        for (const SectionType type : DEFAULT_SUPPORTED_SECTION_TYPES) {
            m_blob_reader.register_section_type_evaluator(std::make_shared<SupportedSectionTypeEvaluator>(type));
        }

        const auto compiler_schedules_instance_evaluator =
            std::make_shared<CompilerScheduleInstanceEvaluator>(m_backend, option_helper);
        m_blob_reader.register_section_instance_evaluator(PredefinedSectionType::ELF_MAIN_SCHEDULE,
                                                          compiler_schedules_instance_evaluator);
        m_blob_reader.register_section_instance_evaluator(PredefinedSectionType::DYNAMIC_SCHEDULE,
                                                          compiler_schedules_instance_evaluator);
    }

    /**
     * @brief Checks if the type and count of sections are valid.
     * @details Expectations:
     *   * The compiled model doesn't contain more than one section of any type
     *   * Either an "ELF main schedule" (with or without init schedules) or a "Dynamic schedule" (without init
     * schedule) exists
     */
    void verify_valid_sections() {
        const bool has_elf_main_schedule = m_blob_reader.has_section_of_type(PredefinedSectionType::ELF_MAIN_SCHEDULE);
        const bool has_init_schedules = m_blob_reader.has_section_of_type(PredefinedSectionType::ELF_INIT_SCHEDULES);
        const bool has_dynamic_schedule = m_blob_reader.has_section_of_type(PredefinedSectionType::DYNAMIC_SCHEDULE);

        OPENVINO_ASSERT(has_elf_main_schedule || has_dynamic_schedule, MISSING_MAIN_SCHEDULE_MESSAGE);
        OPENVINO_ASSERT((has_elf_main_schedule && !has_dynamic_schedule) ||
                            (has_dynamic_schedule && !has_elf_main_schedule && !has_init_schedules),
                        "Found an unsupported combination of compiler schedules within the blob");

        for (const auto& [section_type, count] : m_blob_reader.get_content_summary()) {
            OPENVINO_ASSERT(count <= 1,
                            "Multiple instances of the same section type found inside the blob. This feature is not "
                            "supported yet.");
        }
    }

    /**
     * @brief Decrypts all compiler payloads (main schedule + init schedules if applicable) if:
     *   1. A decryption callback was provided and
     *   2. There is a section indicating the compiler schedules have been encrypted.
     * @throws ov::AssertFailure if the schedules were encrypted, but no decryption callback was provided.
     */
    void decrypt_schedules() override {
        const auto encrypted_schedules_flag_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ENCRYPTED_SCHEDULES_FLAG));
        const bool is_payload_encrypted =
            encrypted_schedules_flag_section ? encrypted_schedules_flag_section->get_flag() : false;
        if (!is_payload_encrypted) {
            m_logger.debug("The compiler payload is NOT encrypted");
            return;
        }

        const bool is_null_decryption = !(m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                          m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt != nullptr);
        OPENVINO_ASSERT(!is_null_decryption, "Blob is encrypted, but no decryption callback was provided!");

        const ov::EncryptionCallbacks encryption_callbacks = m_config.get<CACHE_ENCRYPTION_CALLBACKS>();

        auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::DYNAMIC_SCHEDULE));
        if (dynamic_schedule_section) {
            m_logger.debug("Decrypting the dynamic compiler schedule");
            dynamic_schedule_section->decrypt(encryption_callbacks);
            return;
        }

        auto main_schedule_section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE));
        OPENVINO_ASSERT(main_schedule_section, MISSING_MAIN_SCHEDULE_MESSAGE);

        m_logger.debug("Decrypting the compiler main schedule");
        main_schedule_section->decrypt(encryption_callbacks);

        auto init_schedules_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));
        if (init_schedules_section) {
            m_logger.debug("Decrypting the compiler init schedules");
            init_schedules_section->decrypt(encryption_callbacks);
        }
    }

    ov::Tensor extract_main_schedule() const override {
        const auto main_schedule_section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_MAIN_SCHEDULE));
        if (main_schedule_section) {
            return main_schedule_section->get_schedule();
        }

        const auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::DYNAMIC_SCHEDULE));
        OPENVINO_ASSERT(dynamic_schedule_section, MISSING_MAIN_SCHEDULE_MESSAGE);
        return dynamic_schedule_section->get_schedule();
    }

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override {
        const auto init_schedules_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::ELF_INIT_SCHEDULES));

        return init_schedules_section ? std::make_optional<>(init_schedules_section->get_schedules()) : std::nullopt;
    }

    std::optional<int> extract_batch_size() const override {
        const auto batch_size_section = std::dynamic_pointer_cast<BatchSizeSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::BATCH_SIZE));

        return batch_size_section ? std::make_optional<>(batch_size_section->get_batch_size()) : std::nullopt;
    }

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override {
        const auto io_layouts_section = std::dynamic_pointer_cast<IOLayoutsSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::IO_LAYOUTS));

        return io_layouts_section ? std::make_optional<>(std::make_pair<>(io_layouts_section->get_input_layouts(),
                                                                          io_layouts_section->get_output_layouts()))
                                  : std::nullopt;
    }

    std::optional<uint32_t> extract_compiler_version() const override {
        const auto compiler_version_section = std::dynamic_pointer_cast<CompilerVersionSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::COMPILER_VERSION));

        return compiler_version_section ? std::make_optional<>(compiler_version_section->get_compiler_version())
                                        : std::nullopt;
    }

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override {
        // TODO finish the compat string section
    }

    std::optional<BlobType> extract_blob_type() const override {
        if (m_blob_reader.has_section_of_type(PredefinedSectionType::ELF_MAIN_SCHEDULE)) {
            return BlobType::ELF;
        }

        const auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
            m_blob_reader.retrieve_first_section(PredefinedSectionType::DYNAMIC_SCHEDULE));
        OPENVINO_ASSERT(dynamic_schedule_section, MISSING_MAIN_SCHEDULE_MESSAGE);
        return dynamic_schedule_section->get_blob_type();
    }

    // TODO create blob writer function

    BlobReader m_blob_reader;
};

}  // namespace

namespace intel_npu {

IBlobFormatImporter::IBlobFormatImporter(const std::shared_ptr<const ov::Model>& original_model,
                                         const ov::SoPtr<IEngineBackend>& backend,
                                         const FilteredConfig& config,
                                         const Logger& logger)
    : m_backend(backend),
      m_config(config),
      m_logger(logger),
      m_original_model(original_model) {}

void IBlobFormatImporter::register_compiler_version() {
    std::optional<uint32_t> compiler_version = extract_compiler_version();
    if (compiler_version.has_value()) {
        m_config.update({{ov::intel_npu::compiler_version.name(), std::to_string(compiler_version.value())}});
        m_logger.debug("Imported model was compiled with compiler version: %u.%u",
                       ONEAPI_VERSION_MAJOR(compiler_version.value()),
                       ONEAPI_VERSION_MINOR(compiler_version.value()));
    }
}

std::shared_ptr<IGraph> IBlobFormatImporter::create_graph(const std::string_view network_name,
                                                          const std::string_view device_name,
                                                          const std::shared_ptr<ov::ICore>& core) {
    OV_ITT_TASK_CHAIN(PARSE_AND_CREATE_GRAPH, itt::domains::NPUPlugin, "IBlobFormatImporter", "create_graph");
    m_logger.debug("Creating a graph");

    register_compiler_version();

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "decrypt_schedules");
    decrypt_schedules();

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "extract_main_schedule");
    const ov::Tensor main_schedule = extract_main_schedule();

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "extract_init_schedules");
    const std::optional<std::vector<ov::Tensor>> init_schedules = extract_init_schedules();
    m_batch_size = extract_batch_size();

    update_compiler_type_if_perf_count(m_config, m_backend, device_name);

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "get_parser");
    m_logger.trace("Creating the parser");
    ParserFactory parserFactory;
    auto parser = parserFactory.getParser(m_backend->getInitStructs());

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
                            extract_compiler_compatibility_descriptor(),
                            extract_blob_type());

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

namespace blob_format_importer_factory {

std::unique_ptr<IBlobFormatImporter> create(BlobSource& npu_formatted_blob,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const ov::SoPtr<IEngineBackend>& backend,
                                            const std::shared_ptr<CompilerOptionSupportHelper>& option_helper,
                                            const FilteredConfig& config) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "blob_format_importer_factory::create");
    const size_t input_size = npu_formatted_blob.get_remaining_size();
    OPENVINO_ASSERT(input_size > 0, EMPTY_BLOB_MESSAGE);

    const Logger logger("blob_format_importer_factory", config.get<LOG_LEVEL>());
    if (is_raw_blob) {
        logger.info(BLOB_COMPATIBILITY_SKIPPED_MESSAGE.data());

        logger.debug("Creating a raw blob format import handler using the factory");
        return std::make_unique<RawBlobImporter>(npu_formatted_blob, original_model, backend, config);
    }

    // The V2 format is identified by some magic bytes at the beginning of the input
    OPENVINO_ASSERT(input_size >= MAGIC_BYTES.size(), BLOB_SIZE_SMALLER_THAN_MAGIC);
    const size_t npu_region_start = npu_formatted_blob.tellg();

    std::string blob_magic_bytes(MAGIC_BYTES.size(), 0);
    npu_formatted_blob.read_into_buffer(blob_magic_bytes.data(), MAGIC_BYTES.size());

    if (MAGIC_BYTES == blob_magic_bytes) {
        npu_formatted_blob.seekg(npu_region_start, std::ios::beg);

        logger.debug("Creating a blob format v2 import handler using the factory");
        return std::make_unique<BlobFormatV2Importer>(npu_formatted_blob,
                                                      original_model,
                                                      backend,
                                                      option_helper,
                                                      config);
    }

    // The V1 format is identified by some magic bytes at the end of the input
    npu_formatted_blob.seekg(-static_cast<int>(MAGIC_BYTES.size()), std::ios::end);

    npu_formatted_blob.read_into_buffer(blob_magic_bytes.data(), MAGIC_BYTES.size());
    OPENVINO_ASSERT(MAGIC_BYTES == blob_magic_bytes, "Invalid blob format");

    npu_formatted_blob.seekg(npu_region_start, std::ios::beg);

    logger.debug("Creating a blob format v1 import handler using the factory");
    return std::make_unique<BlobFormatV1Importer>(npu_formatted_blob, original_model, backend, config);
}

}  // namespace blob_format_importer_factory

}  // namespace intel_npu
