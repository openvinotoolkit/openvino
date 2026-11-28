// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_format_importers.hpp"

#include "compiler_option_support_helper.hpp"
#include "hsm_format/batch_size_section.hpp"
#include "hsm_format/compiler_schedule_instance_evaluator.hpp"
#include "hsm_format/compiler_schedules_sections.hpp"
#include "hsm_format/compiler_version_section.hpp"
#include "hsm_format/encrypted_schedules_flag_section.hpp"
#include "hsm_format/io_layouts_section.hpp"
#include "hsm_format/isection.hpp"
#include "hsm_format/read_write.hpp"
#include "hsm_format/runtime_requirements.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/parser_factory.hpp"
#include "intel_npu/config/options.hpp"
#include "intel_npu/utils/tensor_decryption.hpp"
#include "intel_npu/utils/utils.hpp"
#include "metadata.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/allocator.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/util/variant_visitor.hpp"

namespace {

using namespace intel_npu;

constexpr std::string_view BLOB_COMPATIBILITY_SKIPPED_MESSAGE = "Blob compatibility check skipped.";
constexpr std::string_view EMPTY_BLOB_MESSAGE = "The blob provided for import is empty";
constexpr std::string_view MISSING_METADATA_MESSAGE = "The blob is missing the NPU metadata!";
constexpr std::string_view BLOB_SIZE_SMALLER_THAN_MAGIC =
    "Received a blob for import that is not a raw one and its size is smaller than the size of the magic bytes";
constexpr std::string_view EMPTY_COMPILER_PAYLOAD_MESSAGE =
    "The blob provided for import doesn't have any compiler payload";
constexpr std::string_view DECRYPTING_PAYLOAD_MESSAGE = "Decrypting the compiler payload";
constexpr std::string_view NEW_PAGE_ALIGNED_BUFFER_MESSAGE =
    "The compiler payload has been copied into a new, page aligned buffer";
constexpr std::string_view MISSING_MAIN_SCHEDULE_MESSAGE = "The compiler main schedule is missing";

const std::vector<size_t> CONSTANT_NODE_DUMMY_SHAPE{1};

constexpr size_t BRACKETS_SIZE = 2;
constexpr char OPEN_BRACKET = '[';
constexpr char CLOSE_BRACKET = ']';

/**
 * @brief Identifies whether or not the blob source uses the HSM format. If it does, then the base reader for it is
 * returned. "std::nullopt" otherwise.
 */
std::optional<ov::runtime::hsm::Reader> open_hsm_reader(BlobSource& blob_source) {
    std::optional<ov::runtime::hsm::Reader> reader = std::nullopt;

    // The functions provided by OV for creating the reader require either a stream or a buffer object
    std::visit(
        ov::util::VariantVisitor{[&](std::reference_wrapper<std::istream> stream) {
                                     const auto position_before_read = stream.get().tellg();

                                     reader = ov::runtime::hsm::Reader::open(stream);
                                     if (!reader.has_value()) {
                                         // The attempt of reading the HSM format may have put the stream in a bad
                                         // status
                                         stream.get().clear();
                                         stream.get().seekg(position_before_read);
                                     }
                                 },
                                 [&](std::pair<std::reference_wrapper<const ov::Tensor>, size_t> tensor_and_cursor) {
                                     reader = ov::runtime::hsm::Reader::open(
                                         reinterpret_cast<const std::byte*>(tensor_and_cursor.first.get().data()),
                                         tensor_and_cursor.first.get().get_byte_size());
                                 }},
        blob_source.get_source());

    return reader;
}

std::shared_ptr<void> get_shared_object_pointer(BlobSource& blob_source) {
    const auto source = blob_source.get_source();
    if (const auto* tensor_and_cursor =
            std::get_if<std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>(&source)) {
        return ov::get_tensor_impl(tensor_and_cursor->first)._so;
    }

    return nullptr;
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

/**
 * @brief Class used to import a blob that contains only the compiler main schedule.
 */
class RawBlobImporter : public IBlobFormatImporter {
public:
    explicit RawBlobImporter(BlobSource& compiler_main_schedule,
                             const std::shared_ptr<const ov::Model>& original_model,
                             const ov::SoPtr<IEngineBackend>& backend,
                             const Config& config)
        : IBlobFormatImporter(original_model, backend, config, Logger("RawBlobImporter", config.get<LOG_LEVEL>())) {
        const size_t blob_size = compiler_main_schedule.get_remaining_size();

        if (!compiler_main_schedule.is_contiguous()) {
            m_main_schedule = utils::allocate_aligned_tensor(blob_size);
            compiler_main_schedule.read_into_buffer(m_main_schedule.data(), blob_size);

            m_logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data());
            return;
        }

        m_main_schedule = compiler_main_schedule.create_roi_tensor(blob_size);
    }

    /**
     * @note The "raw" format consists of only one main schedule. Therefore, we return a registry containing just one
     * such section.
     */
    BlobSectionsRegistry get_blob_sections_registry() override {
        OPENVINO_ASSERT(m_graph, "Invalid state");
        OPENVINO_ASSERT(m_graph->get_kind() == GraphKind::Weightful,
                        "Only standard, weightful compiler schedules are supported as \"raw blobs\"");

        BlobSectionsRegistry blob_sections_registry;
        blob_sections_registry.register_section(std::make_shared<ELFMainScheduleSection>(m_graph));
        return blob_sections_registry;
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
        utils::decrypt_payload(m_main_schedule, m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt, m_logger);
    }

    ov::Tensor extract_main_schedule() const override {
        return m_main_schedule;
    }

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override {
        return std::nullopt;
    }

    std::optional<int64_t> extract_batch_size() const override {
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
class MetadataFormatImporter : public IBlobFormatImporter {
public:
    explicit MetadataFormatImporter(BlobSource& npu_formatted_blob,
                                    const std::shared_ptr<const ov::Model>& original_model,
                                    const ov::SoPtr<IEngineBackend>& backend,
                                    const Config& config)
        : IBlobFormatImporter(original_model,
                              backend,
                              config,
                              Logger("MetadataFormatImporter", config.get<LOG_LEVEL>())) {
        // Read only the metadata from the source and check if the blob is compatible. Load the blob into memory
        // only if it passes the compatibility checks.
        m_metadata = read_metadata_from(npu_formatted_blob);

        const size_t compiler_payload_size = m_metadata->get_compiler_payload_size();
        OPENVINO_ASSERT(compiler_payload_size > 0, EMPTY_COMPILER_PAYLOAD_MESSAGE);

        if (!npu_formatted_blob.is_contiguous()) {
            m_compiler_payload = utils::allocate_aligned_tensor(compiler_payload_size);
            npu_formatted_blob.read_into_buffer(m_compiler_payload.data(), compiler_payload_size);

            m_logger.info(NEW_PAGE_ALIGNED_BUFFER_MESSAGE.data());
        } else {
            // ROI tensor to skip the NPU plugin metadata
            m_compiler_payload = npu_formatted_blob.create_roi_tensor(compiler_payload_size);
        }
    }

    /**
     * @brief Constructs a blob sections registry that encapsulates the content to be re-exported if requested.
     * @details The sections are built using the content found within the V1 (metadata based) format.
     */
    BlobSectionsRegistry get_blob_sections_registry() override {
        OPENVINO_ASSERT(m_graph, "Invalid state");

        BlobSectionsRegistry blob_sections_registry;

        // Register the compiler schedules
        const bool encryption_enabled = m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                        m_config.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt != nullptr;
        const std::function<std::string(const std::string&)> encryption_callback =
            encryption_enabled ? m_config.get<CACHE_ENCRYPTION_CALLBACKS>().encrypt : nullptr;

        switch (m_graph->get_kind()) {
        case GraphKind::Dynamic: {
            blob_sections_registry.register_section(
                std::make_shared<DynamicScheduleSection>(m_graph, encryption_callback));
            break;
        }
        case GraphKind::Weightless: {
            blob_sections_registry.register_section(
                std::make_shared<ELFInitSchedulesSection>(m_graph, encryption_callback));
            // break omitted on purpose, since in the weightless case we also need to register a main schedule
        }
        case GraphKind::Weightful: {
            blob_sections_registry.register_section(
                std::make_shared<ELFMainScheduleSection>(m_graph, encryption_callback));
            break;
        }
        default: {
            OPENVINO_THROW("Unsupported kind of \"Graph\"");
        }
        }

        // Miscellaneous
        if (m_batch_size.has_value()) {
            blob_sections_registry.register_section(std::make_shared<BatchSizeSection>(m_batch_size.value()));
        }

        const auto layouts = extract_layouts();
        if (layouts.has_value()) {
            blob_sections_registry.register_section(
                std::make_shared<IOLayoutsSection>(layouts->first, layouts->second));
        }

        const auto compiler_version = extract_compiler_version();
        if (compiler_version.has_value()) {
            blob_sections_registry.register_section(std::make_shared<CompilerVersionSection>(compiler_version.value()));
        }

        if (encryption_enabled) {
            blob_sections_registry.register_section(
                std::make_shared<EncryptedSchedulesFlagSection>(encryption_enabled));
        }

        return blob_sections_registry;
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
        utils::decrypt_payload(m_compiler_payload, m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt, m_logger);
    }

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

    std::optional<int64_t> extract_batch_size() const override {
        const std::optional<int64_t> batch_size = m_metadata->get_batch_size();
        if (batch_size.has_value()) {
            m_logger.debug("Extracted batch size: %lld", static_cast<long long>(batch_size.value()));
        }
        return batch_size;
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
class HSMFormatImporter : public IBlobFormatImporter {
public:
    /**
     * @param base_reader The HSM reader provided by OV, used to extract all sections as binary buffers before parsing
     * them.
     * @param blob_so If @ref import_model(ov::Tensor) was called, then this object is a reference towards the tensor
     * that we want to keep alive. Otherwise, it's just a nullptr.
     */
    explicit HSMFormatImporter(ov::runtime::hsm::Reader& base_reader,
                               const std::shared_ptr<void> blob_so,
                               const std::shared_ptr<const ov::Model>& original_model,
                               const ov::SoPtr<IEngineBackend>& backend,
                               const std::shared_ptr<CompilerOptionSupportHelper>& option_helper,
                               const Config& config)
        : IBlobFormatImporter(original_model, backend, config, Logger("HSMFormatImporter", config.get<LOG_LEVEL>())),
          m_blob_so(blob_so) {
        std::tie(m_blob_sections_registry, m_runtime_requirements) =
            hsm::read_from(base_reader, get_section_readers(), {}, create_instance_evaluators(option_helper));

        if (!m_runtime_requirements.has_value()) {
            // This runtime requirements object contains more than the compiler requirements. But we need it only for
            // extracting the compiler ones.
            m_logger.warning(
                "Missing compiler runtime requirements while importing the compiled model. No compiler runtime "
                "requirements will be provided upon querying the compiled model property or exporting the model");
        }

        verify_valid_sections();
    }

    /**
     * @brief Updates the state of the compiler schedules sections, and then returns the registry.
     */
    BlobSectionsRegistry get_blob_sections_registry() override {
        OPENVINO_ASSERT(m_graph, "Invalid state");

        // Moving forward, the "Graph" object will manage the ownership of the compiler schedules
        auto elf_main_schedule_section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_MAIN_SCHEDULE));

        if (elf_main_schedule_section) {
            auto init_schedules_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
                m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_INIT_SCHEDULES));

            elf_main_schedule_section->set_graph(m_graph);
            if (init_schedules_section) {
                init_schedules_section->set_graph(m_graph);
            }
        } else {
            auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
                m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::DYNAMIC_SCHEDULE));
            dynamic_schedule_section->set_graph(m_graph);
        }

        return m_blob_sections_registry;
    }

private:
    /**
     * @returns All blob section readers known to the plugin.
     */
    std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>> get_section_readers()
        const {
        return {
            {SectionTypeCode::RUNTIME_REQUIREMENTS, RuntimeRequirementsSection::decode},
            {SectionTypeCode::ELF_MAIN_SCHEDULE, ELFMainScheduleSection::decode},
            {SectionTypeCode::ELF_INIT_SCHEDULES, ELFInitSchedulesSection::decode},
            {SectionTypeCode::DYNAMIC_SCHEDULE, DynamicScheduleSection::decode},
            {SectionTypeCode::IO_LAYOUTS, IOLayoutsSection::decode},
            {SectionTypeCode::BATCH_SIZE, BatchSizeSection::decode},
            {SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG, EncryptedSchedulesFlagSection::decode},
            {SectionTypeCode::COMPILER_VERSION, CompilerVersionSection::decode},
        };
    }

    /**
     * @brief Creates instance evaluators for all sections that require them.
     */
    std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>> create_instance_evaluators(
        const std::shared_ptr<CompilerOptionSupportHelper>& option_helper) const {
        const auto compiler_schedules_instance_evaluator =
            CompilerScheduleInstanceEvaluator::get_instance(m_backend, option_helper);
        return {
            {SectionTypeCode::ELF_MAIN_SCHEDULE, compiler_schedules_instance_evaluator},
            {SectionTypeCode::DYNAMIC_SCHEDULE, compiler_schedules_instance_evaluator},
        };
    }

    /**
     * @brief Checks if the type and count of sections are aligned with the expectations of the format.
     * @details Expectations:
     *   * The compiled model doesn't contain more than one section of any type
     *   * Either an @ref ELFMainScheduleSection (with or without init schedules) or a @ref DynamicScheduleSection
     * (without any init schedule) exists
     */
    void verify_valid_sections() {
        const bool has_elf_main_schedule =
            m_blob_sections_registry.has_section_of_type(SectionTypeCode::ELF_MAIN_SCHEDULE);
        const bool has_init_schedules =
            m_blob_sections_registry.has_section_of_type(SectionTypeCode::ELF_INIT_SCHEDULES);
        const bool has_dynamic_schedule =
            m_blob_sections_registry.has_section_of_type(SectionTypeCode::DYNAMIC_SCHEDULE);

        OPENVINO_ASSERT(has_elf_main_schedule || has_dynamic_schedule, MISSING_MAIN_SCHEDULE_MESSAGE);
        OPENVINO_ASSERT((has_elf_main_schedule && !has_dynamic_schedule) ||
                            (has_dynamic_schedule && !has_elf_main_schedule && !has_init_schedules),
                        "Found an unsupported combination of compiler schedules within the blob");

        for (const auto& [section_type, count] : m_blob_sections_registry.get_content_summary()) {
            OPENVINO_ASSERT(count <= 1,
                            "Multiple instances of the same section type found inside the blob. This feature is not "
                            "supported yet.");
        }
    }

    /**
     * @brief Decrypts all compiler payloads (either a dynamic schedule or main schedule + init schedules if applicable)
     * if:
     *   1. A decryption callback was provided and
     *   2. There is a section ( @ref EncryptedSchedulesFlagSection ) indicating the compiler schedules have been
     * encrypted.
     * @throws ov::AssertFailure if the schedules were encrypted, but no decryption callback was provided.
     */
    void decrypt_schedules() override {
        const auto encrypted_schedules_flag_section = std::dynamic_pointer_cast<EncryptedSchedulesFlagSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG));
        const bool is_payload_encrypted =
            encrypted_schedules_flag_section ? encrypted_schedules_flag_section->get_flag() : false;
        if (!is_payload_encrypted) {
            m_logger.debug("The compiler payload is NOT encrypted");
            return;
        }

        const bool is_null_decryption = !(m_config.has(CACHE_ENCRYPTION_CALLBACKS::key().data()) &&
                                          m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt != nullptr);
        OPENVINO_ASSERT(!is_null_decryption, "Blob is encrypted, but no decryption callback was provided!");

        const std::function<std::string(const std::string&)> decryption_callback =
            m_config.get<CACHE_ENCRYPTION_CALLBACKS>().decrypt;

        auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::DYNAMIC_SCHEDULE));
        if (dynamic_schedule_section) {
            m_logger.debug("Decrypting the dynamic compiler schedule");
            dynamic_schedule_section->decrypt(decryption_callback);
            return;
        }

        auto main_schedule_section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_MAIN_SCHEDULE));
        OPENVINO_ASSERT(main_schedule_section, MISSING_MAIN_SCHEDULE_MESSAGE);

        m_logger.debug("Decrypting the compiler main schedule");
        main_schedule_section->decrypt(decryption_callback);

        auto init_schedules_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_INIT_SCHEDULES));
        if (init_schedules_section) {
            m_logger.debug("Decrypting the compiler init schedules");
            init_schedules_section->decrypt(decryption_callback);
        }
    }

    ov::Tensor extract_main_schedule() const override {
        const auto main_schedule_section = std::dynamic_pointer_cast<ELFMainScheduleSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_MAIN_SCHEDULE));
        if (main_schedule_section) {
            // If `import_model(ov::Tensor)` was called, then the shared object corresponds to the tensor that we want
            // to keep alive. Otherwise, it's just a nullptr.
            return ov::Tensor(main_schedule_section->get_schedule(), m_blob_so);
        }

        const auto dynamic_schedule_section = std::dynamic_pointer_cast<DynamicScheduleSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::DYNAMIC_SCHEDULE));
        OPENVINO_ASSERT(dynamic_schedule_section, MISSING_MAIN_SCHEDULE_MESSAGE);
        return dynamic_schedule_section->get_schedule();
    }

    std::optional<std::vector<ov::Tensor>> extract_init_schedules() const override {
        const auto init_schedules_section = std::dynamic_pointer_cast<ELFInitSchedulesSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::ELF_INIT_SCHEDULES));

        return init_schedules_section ? std::make_optional<>(init_schedules_section->get_schedules()) : std::nullopt;
    }

    std::optional<int64_t> extract_batch_size() const override {
        const auto batch_size_section = std::dynamic_pointer_cast<BatchSizeSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::BATCH_SIZE));

        return batch_size_section ? std::make_optional<>(batch_size_section->get_batch_size()) : std::nullopt;
    }

    std::optional<std::pair<std::vector<ov::Layout>, std::vector<ov::Layout>>> extract_layouts() const override {
        const auto io_layouts_section = std::dynamic_pointer_cast<IOLayoutsSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::IO_LAYOUTS));

        return io_layouts_section ? std::make_optional<>(std::make_pair<>(io_layouts_section->get_input_layouts(),
                                                                          io_layouts_section->get_output_layouts()))
                                  : std::nullopt;
    }

    std::optional<uint32_t> extract_compiler_version() const override {
        const auto compiler_version_section = std::dynamic_pointer_cast<CompilerVersionSection>(
            m_blob_sections_registry.retrieve_unique_section_of_type(SectionTypeCode::COMPILER_VERSION));

        return compiler_version_section ? std::make_optional<>(compiler_version_section->get_compiler_version())
                                        : std::nullopt;
    }

    std::optional<std::string> extract_compiler_compatibility_descriptor() const override {
        if (!m_runtime_requirements.has_value()) {
            return std::nullopt;
        }

        // Look for the runtime requirements of the ELF main/dynamic schedule. The checks performed previously
        // should already guarantee that only one instance of these section types exists
        const std::map<SectionID, std::string> sections_requirements =
            m_runtime_requirements->get_sections_requirements();
        const std::unordered_map<SectionID, SectionType> section_id_to_type_mapping =
            m_runtime_requirements->get_section_id_to_type_mapping();

        for (const auto& [section_id, section_requirements] : sections_requirements) {
            if (section_id_to_type_mapping.at(section_id) == SectionType(SectionTypeCode::ELF_MAIN_SCHEDULE) ||
                section_id_to_type_mapping.at(section_id) == SectionType(SectionTypeCode::DYNAMIC_SCHEDULE)) {
                // Strip the "[]" if applicable
                const size_t section_requirements_size = section_requirements.size();
                if (section_requirements_size >= BRACKETS_SIZE && section_requirements.at(0) == OPEN_BRACKET &&
                    section_requirements.at(section_requirements_size - 1) == CLOSE_BRACKET) {
                    return section_requirements.substr(1, section_requirements_size - 2);
                }
                return section_requirements;
            }
        }

        return std::nullopt;
    }

    std::optional<BlobType> extract_blob_type() const override {
        if (m_blob_sections_registry.has_section_of_type(SectionTypeCode::ELF_MAIN_SCHEDULE)) {
            return BlobType::ELF;
        }

        OPENVINO_ASSERT(m_blob_sections_registry.has_section_of_type(SectionTypeCode::DYNAMIC_SCHEDULE));
        return BlobType::BYTECODE;
    }

    BlobSectionsRegistry m_blob_sections_registry;
    std::optional<RuntimeRequirements> m_runtime_requirements;
    std::shared_ptr<void> m_blob_so;
};

}  // namespace

namespace intel_npu {
IBlobFormatImporter::IBlobFormatImporter(const std::shared_ptr<const ov::Model>& original_model,
                                         const ov::SoPtr<IEngineBackend>& backend,
                                         const Config& config,
                                         const Logger& logger)
    : m_backend(backend),
      m_config(config),
      m_logger(logger),
      m_original_model(original_model) {}

void IBlobFormatImporter::register_compiler_version() {
    std::optional<uint32_t> compiler_version = extract_compiler_version();
    if (compiler_version.has_value()) {
        m_config.update(ov::intel_npu::compiler_version.name(), std::to_string(compiler_version.value()));
        m_logger.debug("Imported model was compiled with compiler version: %u.%u",
                       ONEAPI_VERSION_MAJOR(compiler_version.value()),
                       ONEAPI_VERSION_MINOR(compiler_version.value()));
    }
}

std::shared_ptr<IGraph> IBlobFormatImporter::create_graph(const std::string_view network_name,
                                                          const std::shared_ptr<ov::ICore>& core,
                                                          const VCLFunctionTableProvider& vclFunctionsProvider) {
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

    OPENVINO_ASSERT(m_backend);

    OV_ITT_TASK_NEXT(PARSE_AND_CREATE_GRAPH, "get_parser");
    m_logger.trace("Creating the parser");
    ParserFactory parserFactory;
    auto parser =
        parserFactory.getParser(m_backend->getInitStructs(), vclFunctionsProvider, m_config.get<PERF_COUNT>());

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
    blob_format_importer_utils::validate_main_schedule_descriptors(m_graph->get_metadata());
    blob_format_importer_utils::validate_batch_size(m_graph->get_metadata(), m_batch_size);
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
                                layouts.has_value() ? std::make_optional(layouts->first) : std::nullopt,
                                layouts.has_value() ? std::make_optional(layouts->second) : std::nullopt);
}

Config IBlobFormatImporter::get_config() const {
    return m_config;
}

namespace blob_format_importer_factory {

std::unique_ptr<IBlobFormatImporter> create(BlobSource& blob_source,
                                            const bool is_raw_blob,
                                            const std::shared_ptr<const ov::Model>& original_model,
                                            const ov::SoPtr<IEngineBackend>& backend,
                                            const std::shared_ptr<CompilerOptionSupportHelper>& option_helper,
                                            const Config& config) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "blob_format_importer_factory::create");
    const Logger logger("blob_format_importer_factory", config.get<LOG_LEVEL>());

    const size_t input_size = blob_source.get_remaining_size();
    OPENVINO_ASSERT(input_size > 0, EMPTY_BLOB_MESSAGE);

    if (is_raw_blob) {
        logger.info(BLOB_COMPATIBILITY_SKIPPED_MESSAGE.data());

        logger.debug("Creating a raw blob format import handler using the factory");
        return std::make_unique<RawBlobImporter>(blob_source, original_model, backend, config);
    }

    std::optional<ov::runtime::hsm::Reader> reader = open_hsm_reader(blob_source);
    if (reader.has_value()) {
        logger.debug("Creating a blob format v2 import handler using the factory");
        return std::make_unique<HSMFormatImporter>(reader.value(),
                                                   get_shared_object_pointer(blob_source),
                                                   original_model,
                                                   backend,
                                                   option_helper,
                                                   config);
    }

    // The V1 format is identified by some magic bytes at the end of the input
    OPENVINO_ASSERT(input_size >= MAGIC_BYTES.size(), BLOB_SIZE_SMALLER_THAN_MAGIC);

    const size_t compiler_payload_beginning = blob_source.tellg();
    blob_source.seekg(-static_cast<int>(MAGIC_BYTES.size()), std::ios::end);

    std::string blob_magic_bytes(MAGIC_BYTES.size(), 0);
    blob_source.read_into_buffer(blob_magic_bytes.data(), MAGIC_BYTES.size());

    OPENVINO_ASSERT(MAGIC_BYTES == blob_magic_bytes, MISSING_METADATA_MESSAGE);
    blob_source.seekg(compiler_payload_beginning, std::ios::beg);

    logger.debug("Creating a blob format v1 import handler using the factory");
    return std::make_unique<MetadataFormatImporter>(blob_source, original_model, backend, config);
}

}  // namespace blob_format_importer_factory
namespace blob_format_importer_utils {

void validate_main_schedule_descriptors(const NetworkMetadata& metadata) {
    auto check_descriptors = [](const std::vector<IODescriptor>& descriptors) {
        for (const IODescriptor& descriptor : descriptors) {
            OPENVINO_ASSERT(!descriptor.isInitInputWeights && !descriptor.isInitOutputWeights,
                            "The I/O descriptor \"",
                            descriptor.nameFromCompiler,
                            "\" of the main schedule is marked as an init schedule weights descriptor");
        }
    };

    check_descriptors(metadata.inputs);
    check_descriptors(metadata.outputs);
}

void validate_batch_size(const NetworkMetadata& metadata, const std::optional<int64_t>& batch_size) {
    if (!batch_size.has_value()) {
        return;
    }

    // Any non-positive value other than the dynamic batch marker cannot be produced by a valid export.
    OPENVINO_ASSERT(batch_size.value() > 0 || batch_size.value() == utils::DYNAMIC_BATCH_SIZE,
                    "The blob metadata declares an invalid batch size: ",
                    batch_size.value());

    // The schedule is compiled with a batch of 1 for both a static and a dynamic batch handled by the plugin. The
    // plugin runs it once per batch slice, advancing the I/O buffers by one slice per execution, so a descriptor which
    // is not batched would be written out of bounds when the actual batch is greater than 1.
    const std::string declared_batch_size = batch_size.value() == utils::DYNAMIC_BATCH_SIZE
                                                ? "a dynamic batch size"
                                                : "a batch size of " + std::to_string(batch_size.value());

    auto check_descriptors = [&declared_batch_size](const std::vector<IODescriptor>& descriptors, const char* kind) {
        for (const IODescriptor& descriptor : descriptors) {
            // States and shape tensors cannot be combined with the plugin batching: models with variables or with
            // dynamic dimensions other than the batch are compiled by the compiler as a whole
            OPENVINO_ASSERT(!descriptor.isStateInput && !descriptor.isStateOutput && !descriptor.isShapeTensor,
                            "The blob metadata declares ",
                            declared_batch_size,
                            ", but the ",
                            kind,
                            " \"",
                            descriptor.nameFromCompiler,
                            "\" is a ",
                            descriptor.isShapeTensor ? "shape tensor" : "state",
                            ", which is not supported when batching is handled by the plugin");

            // The weights of the main schedule have nothing to do with the batch. The init schedule weights are
            // rejected by "validate_main_schedule_descriptors".
            if (descriptor.isMainInputWeights) {
                continue;
            }

            const ov::PartialShape& shape = descriptor.shapeFromCompiler;
            // Every descriptor must have a static batch of 1: a dynamic rank, a scalar or a dynamic batch dimension
            // means there is no batch slice to advance over. Dynamic dimensions other than the batch are accepted.
            OPENVINO_ASSERT(shape.rank().is_static() && shape.size() > 0 && shape[utils::BATCH_AXIS].is_static() &&
                                static_cast<size_t>(shape[utils::BATCH_AXIS].get_length()) == utils::DEFAULT_BATCH_SIZE,
                            "The blob metadata declares ",
                            declared_batch_size,
                            ", but the ",
                            kind,
                            " \"",
                            descriptor.nameFromCompiler,
                            "\" was compiled with the shape ",
                            shape.to_string(),
                            ". Batching on the plugin requires a batch of 1 in the compiled schedule");
        }
    };

    check_descriptors(metadata.inputs, "input");
    check_descriptors(metadata.outputs, "output");
}

}  // namespace blob_format_importer_utils

}  // namespace intel_npu
