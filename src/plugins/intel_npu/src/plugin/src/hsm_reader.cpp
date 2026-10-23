// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "hsm_reader.hpp"

#include "intel_npu/common/itt.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"

using namespace intel_npu;

namespace {

constexpr std::string_view PARSE_FAILURE_MESSAGE = "Failed to parse the mandatory section ";

const ov::runtime::hsm::SectionTag RUNTIME_REQUIREMENTS_TAG =
    ov::runtime::hsm::SectionTag::make(static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS), false);

// TODO comment mirrors writer conversion
SectionID tag_reserved_bytes_to_section_id(const std::array<uint8_t, 4>& reserved_bytes) {
    return SectionID(reserved_bytes.at(0) + (reserved_bytes.at(1) << 8));
}

}  // namespace

namespace intel_npu::hsm {

// TODO ITT here
Reader::Reader(
    const std::unordered_map<SectionType, ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>>>& readers,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators)
    : m_readers(readers),
      m_type_evaluators(type_evaluators),
      m_instance_evaluators(instance_evaluators),
      m_logger("HSMReader", Logger::global().level()) {}

BlobSectionsRegistry Reader::read(ov::runtime::hsm::Reader& base_reader) const {
    // TODO dedicated tags instead of sectin types?
    std::optional<RuntimeRequirements> runtime_requirements;
    const size_t n_runtime_requirements =
        base_reader.count(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                          RUNTIME_REQUIREMENTS_TAG);
    OPENVINO_ASSERT(n_runtime_requirements < 2, "Found more than one runtime requirements section");

    if (n_runtime_requirements) {
        OPENVINO_ASSERT(m_readers.count(SectionTypeCode::RUNTIME_REQUIREMENTS), "No reader found for the manifest");
        const auto decode_result =
            base_reader.decode(static_cast<ov::runtime::hsm::DeviceId>(ov::runtime::hsm::Devices::npu),
                               RUNTIME_REQUIREMENTS_TAG,
                               m_readers.at(SectionTypeCode::RUNTIME_REQUIREMENTS));
        OPENVINO_ASSERT(decode_result.has_value());
        const auto runtime_requirements_section =
            std::dynamic_pointer_cast<RuntimeRequirementsSection>(decode_result.value());
        OPENVINO_ASSERT(runtime_requirements_section);

        runtime_requirements = runtime_requirements_section->get_runtime_requirements();
        OPENVINO_ASSERT(runtime_requirements->get_compatibility_check_result(m_type_evaluators, m_instance_evaluators),
                        "The imported model is not compatible");
        m_logger.debug("Runtime requirements evaluation passed");
    } else {
        m_logger.warning(
            "The runtime requirements section was not found within the manifest. Proceeding without performing any "
            "compatibility checks");
    }

    BlobSectionsRegistry registry;

    // TODO veridy everything important has been imported from blobwriter/reader
    for (const ov::runtime::hsm::ManifestEntry& manifest_entry : base_reader.entries()) {
        if (manifest_entry.tag.id() == static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS)) {
            // Already parsed
            continue;
        }

        const SectionType section_type(static_cast<SectionTypeCode>(manifest_entry.tag.id()));
        const SectionID section_id(tag_reserved_bytes_to_section_id(manifest_entry.tag_reserved));
        const std::string debug_name = section_type_and_id_to_string(section_type, section_id);

        m_logger.debug("Parsing the section %s", debug_name.data());

        // The section is considered for parsing only if the BlobReader has a reader registered for its type
        if (!m_readers.count(section_type)) {
            m_logger.debug("No section reader found for section %s. Skipping", debug_name.data());
            continue;
        }
        m_logger.trace("Found a reader for section %s", debug_name.data());

        if (!runtime_requirements.has_value()) {
            // No runtime requirements found. All sections are considered mandatory
            m_logger.trace("Parsing the mandatory section %s", debug_name.data());

            const std::optional<std::shared_ptr<ISection>> parsed_section =
                base_reader.decode(manifest_entry.device, manifest_entry.tag, m_readers.at(section_type));
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
            base_reader.decode(manifest_entry.device, manifest_entry.tag, m_readers.at(section_type));
        if (parsed_section.has_value()) {
            registry.register_section(parsed_section.value(), section_id);
        }

        if (type_evaluation_result.has_value() && type_evaluation_result.value()) {
            if (instance_evaluation_result.has_value() && instance_evaluation_result.value()) {
                // Case 1
                // TODO update logs
                m_logger.trace("Parsing mandatory section %s", debug_name.data());

                const std::optional<std::shared_ptr<ISection>> parsed_section =
                    base_reader.decode(manifest_entry.device, manifest_entry.tag, m_readers.at(section_type));
                OPENVINO_ASSERT(parsed_section.has_value(), PARSE_FAILURE_MESSAGE, debug_name);
                registry.register_section(parsed_section.value(), section_id);
            } else if (instance_evaluation_result.has_value() && !instance_evaluation_result.value()) {
                // Case 2
                m_logger.debug(
                    "The parsing of section ID %s has been skipped. The section type instance is not supported",
                    debug_name.data());
            } else {
                // Case 3
                m_logger.trace("The section type instance corresponding to section ID %s was not evaluated",
                               debug_name.data());

                const std::optional<std::shared_ptr<ISection>> parsed_section =
                    base_reader.decode(manifest_entry.device, manifest_entry.tag, m_readers.at(section_type));
                OPENVINO_ASSERT(parsed_section.has_value(), PARSE_FAILURE_MESSAGE, debug_name);
                if (parsed_section.has_value()) {
                    registry.register_section(parsed_section.value(), section_id);
                } else {
                    m_logger.warning("The parsing of optional section %s has failed.", debug_name.data());
                }
            }
        } else if (type_evaluation_result.has_value() && !type_evaluation_result.value()) {
            // Case 4
            m_logger.debug("The parsing of section %s has been skipped. The section type is not supported",
                           debug_name.data());
        } else {
            // Case 5
            m_logger.trace("The section type of %s was not evaluated", debug_name.data());
            OPENVINO_ASSERT(
                !instance_evaluation_result.has_value(),
                "Found a section type instance that was evaluated without evaluating the section type first");

            // TODO refactor opportunities here?
            const std::optional<std::shared_ptr<ISection>> parsed_section =
                base_reader.decode(manifest_entry.device, manifest_entry.tag, m_readers.at(section_type));
            OPENVINO_ASSERT(parsed_section.has_value(), PARSE_FAILURE_MESSAGE, debug_name);
            if (parsed_section.has_value()) {
                registry.register_section(parsed_section.value(), section_id);
            } else {
                m_logger.warning("The parsing of optional section %s has failed.", debug_name.data());
            }
        }
    }
}

}  // namespace intel_npu::hsm
