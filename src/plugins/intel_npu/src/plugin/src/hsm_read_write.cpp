// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_read_write.hpp"

#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

using namespace intel_npu;

namespace {

constexpr std::string_view PARSE_FAILURE_MESSAGE = "Failed to parse the mandatory section ";
constexpr std::string_view MANDATORY_READER_MISSING_MESSAGE = "No section reader found for mandatory section ";
constexpr std::string_view OPTIONAL_READER_MISSING_MESSAGE = "No section reader found for optional section %s";
constexpr std::string_view OPTIONAL_SECTION_PARSING_FAILED_MESSAGE = "The parsing of optional section %s has failed.";

const ov::runtime::hsm::SectionTag RUNTIME_REQUIREMENTS_TAG =
    ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS), false);

// TODO comment mirrors writer conversion
SectionID tag_reserved_bytes_to_section_id(const std::array<uint8_t, 4>& reserved_bytes) {
    return SectionID(reserved_bytes.at(0) + (reserved_bytes.at(1) << 8));
}

ov::runtime::hsm::SectionTagReserved build_section_tag(const std::shared_ptr<ISection>& section) {
    OPENVINO_ASSERT(section->get_id().has_value());
    const SectionID id = section->get_id().value();

    const ov::runtime::hsm::SectionTag tag =
        ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(section->get_type().get_code()), section->is_inline());
    std::array<uint8_t, 4> reserved_bytes{static_cast<uint8_t>(static_cast<uint8_t>(id.get_id()) & 0xFF),
                                          static_cast<uint8_t>(static_cast<uint8_t>(id.get_id() >> 8) & 0xFF),
                                          0,
                                          0};
    return ov::runtime::hsm::SectionTagReserved(tag, reserved_bytes);
}

RuntimeRequirements parse_and_evaluate_runtime_requirements(
    ov::runtime::hsm::Reader& base_reader,
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& section_readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators,
    const Logger& logger) {
    OPENVINO_ASSERT(section_readers.count(SectionTypeCode::RUNTIME_REQUIREMENTS),
                    "No reader found for the runtime requirements section");
    const auto decode_result =
        base_reader.decode(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                           RUNTIME_REQUIREMENTS_TAG.id(),
                           section_readers.at(SectionTypeCode::RUNTIME_REQUIREMENTS));
    OPENVINO_ASSERT(decode_result.has_value());
    const auto runtime_requirements_section =
        std::dynamic_pointer_cast<RuntimeRequirementsSection>(decode_result.value());
    OPENVINO_ASSERT(runtime_requirements_section);

    RuntimeRequirements runtime_requirements = runtime_requirements_section->get_runtime_requirements();
    OPENVINO_ASSERT(runtime_requirements.get_compatibility_check_result(type_evaluators, instance_evaluators),
                    "The imported model is not compatible");
    logger.debug("Runtime requirements evaluation passed");
    return runtime_requirements;
}

}  // namespace

namespace intel_npu::hsm {

// TODO ITT traces

BlobSectionsRegistry read_from(
    ov::runtime::hsm::Reader& base_reader,
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& section_readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators) {
    const Logger logger("HSMReader", Logger::global().level());
    logger.debug("Parsing the HSM container");

    // TODO dedicated tags instead of sectin types?
    std::optional<RuntimeRequirements> runtime_requirements;
    const size_t n_runtime_requirements =
        base_reader.count(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                          RUNTIME_REQUIREMENTS_TAG.id());
    OPENVINO_ASSERT(n_runtime_requirements < 2, "Found more than one runtime requirements section");

    if (n_runtime_requirements) {
        runtime_requirements = parse_and_evaluate_runtime_requirements(base_reader,
                                                                       section_readers,
                                                                       type_evaluators,
                                                                       instance_evaluators,
                                                                       logger);
    } else {
        logger.warning(
            "The runtime requirements section was not found within the manifest. Proceeding without performing any "
            "compatibility checks");
    }

    BlobSectionsRegistry registry;

    for (const ov::runtime::hsm::ManifestEntry& manifest_entry : base_reader.entries()) {
        if (manifest_entry.tag.id() == static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS)) {
            // Already parsed
            continue;
        }

        const SectionType section_type(static_cast<SectionTypeCode>(manifest_entry.tag.id()));
        const SectionID section_id(tag_reserved_bytes_to_section_id(manifest_entry.tag_reserved));
        const std::string debug_name = section_type_and_id_to_string(section_type, section_id);

        logger.debug("Parsing the section %s", debug_name.data());

        // TODO test the negative branch as well
        if (!runtime_requirements.has_value()) {
            // No runtime requirements found. All sections are considered mandatory
            logger.trace("Parsing the mandatory section %s", debug_name.data());
            OPENVINO_ASSERT(section_readers.count(section_type), MANDATORY_READER_MISSING_MESSAGE, debug_name);

            const std::optional<std::shared_ptr<ISection>> parsed_section =
                base_reader.decode(manifest_entry.device, manifest_entry.tag.id(), section_readers.at(section_type));
            OPENVINO_ASSERT(parsed_section.has_value(), PARSE_FAILURE_MESSAGE, debug_name);
            registry.register_section(parsed_section.value(), section_id);
            continue;
        }

        // How the section will be handled depends on the type & instance evaluators:
        //  * Case 1: evaluated & supported type, evaluated & supported instance - the section is mandatory. It is read
        //  without a try-catch block.
        //  * Case 2: evaluated & supported type, evaluated & unsupported instance - the section is unsupported. It's
        //  parsing is skipped.
        //  * Case 3: evaluated & supported type, unevaluated instance - the section is optional. It is read within a
        //  try-catch block.
        //  * Case 4: evaluated & unsupported type - the section is unsupported. It's parsing is skipped.
        //  * Case 5: unevaluated type, unevaluated instance - the section is optional. It is read within a try-catch
        //  block.

        const std::optional<bool> type_evaluation_result =
            runtime_requirements->get_type_evaluation_result(section_type);
        const std::optional<bool> instance_evaluation_result =
            runtime_requirements->get_instance_evaluation_result(section_id);

        const std::optional<std::shared_ptr<ISection>> parsed_section =
            base_reader.decode(manifest_entry.device, manifest_entry.tag.id(), section_readers.at(section_type));
        if (parsed_section.has_value()) {
            registry.register_section(parsed_section.value(), section_id);
        }

        if (type_evaluation_result.has_value() && type_evaluation_result.value()) {
            if (instance_evaluation_result.has_value() && instance_evaluation_result.value()) {
                // Case 1
                logger.trace("Parsing the mandatory section %s", debug_name.data());
                OPENVINO_ASSERT(section_readers.count(section_type), MANDATORY_READER_MISSING_MESSAGE, debug_name);

                const std::optional<std::shared_ptr<ISection>> parsed_section =
                    base_reader.decode(manifest_entry.device,
                                       manifest_entry.tag.id(),
                                       section_readers.at(section_type));
                OPENVINO_ASSERT(parsed_section.has_value(), PARSE_FAILURE_MESSAGE, debug_name);
                registry.register_section(parsed_section.value(), section_id);
            } else if (instance_evaluation_result.has_value() && !instance_evaluation_result.value()) {
                // Case 2
                logger.debug("The parsing of section %s has been skipped. The section instance is not supported",
                             debug_name.data());
            } else {
                // Case 3
                logger.trace("The section instance %s was not evaluated", debug_name.data());
                logger.trace("Parsing the optional section %s", debug_name.data());
                if (!section_readers.count(section_type)) {
                    logger.debug(OPTIONAL_READER_MISSING_MESSAGE.data(), debug_name.data());
                    logger.warning(OPTIONAL_SECTION_PARSING_FAILED_MESSAGE.data(), debug_name.data());
                    continue;
                }

                const std::optional<std::shared_ptr<ISection>> parsed_section =
                    base_reader.decode(manifest_entry.device,
                                       manifest_entry.tag.id(),
                                       section_readers.at(section_type));
                if (parsed_section.has_value()) {
                    registry.register_section(parsed_section.value(), section_id);
                } else {
                    logger.warning(OPTIONAL_SECTION_PARSING_FAILED_MESSAGE.data(), debug_name.data());
                }
            }
        } else if (type_evaluation_result.has_value() && !type_evaluation_result.value()) {
            // Case 4
            logger.debug("The parsing of section %s has been skipped. The section type is not supported",
                         debug_name.data());
        } else {
            // Case 5
            logger.trace("The section type of %s was not evaluated", debug_name.data());
            logger.trace("Parsing the optional section %s", debug_name.data());
            OPENVINO_ASSERT(!instance_evaluation_result.has_value(),
                            "Found a section instance that was evaluated without evaluating the section type first");
            if (!section_readers.count(section_type)) {
                logger.debug(OPTIONAL_READER_MISSING_MESSAGE.data(), debug_name.data());
                logger.warning(OPTIONAL_SECTION_PARSING_FAILED_MESSAGE.data(), debug_name.data());
                continue;
            }

            const std::optional<std::shared_ptr<ISection>> parsed_section =
                base_reader.decode(manifest_entry.device, manifest_entry.tag.id(), section_readers.at(section_type));
            if (parsed_section.has_value()) {
                registry.register_section(parsed_section.value(), section_id);
            } else {
                logger.warning(OPTIONAL_SECTION_PARSING_FAILED_MESSAGE.data(), debug_name.data());
            }
        }
    }

    return registry;
}

void write_to(std::ostream& stream, const BlobSectionsRegistry& blob_sections_registry) {
    const Logger logger("HSMWriter", Logger::global().level());
    logger.debug("Writing into stream using the HSM format");

    auto deferred_writer = ov::runtime::hsm::DeferredWriter::open(stream);
    OPENVINO_ASSERT(deferred_writer.has_value(), "A blob writer could not be created. The stream may be invalid");

    // TODO add common sections when available
    const auto runtime_requirements_section = blob_sections_registry.build_runtime_requirements_section();
    deferred_writer->add_section(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                                 build_section_tag(runtime_requirements_section),
                                 runtime_requirements_section,
                                 runtime_requirements_section->get_alignment());

    std::queue<std::shared_ptr<ISection>> write_queue = blob_sections_registry.get_write_queue();
    while (!write_queue.empty()) {
        const std::shared_ptr<ISection>& section = write_queue.front();
        write_queue.pop();
        deferred_writer->add_section(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                                     build_section_tag(section),
                                     section,
                                     section->get_alignment());
    }

    deferred_writer->finalize();
}

}  // namespace intel_npu::hsm
