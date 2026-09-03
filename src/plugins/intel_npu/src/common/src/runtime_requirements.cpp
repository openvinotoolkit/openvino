// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/major_minor_version.hpp"
#include "intel_npu/compat_string_parser.hpp"

namespace {

constexpr std::string_view VERSION_KEY = "version";
constexpr std::string_view CRE_KEY = "cre";
constexpr char REQUIREMENTS_SEPARATOR = ';';
constexpr char KEY_VALUE_SEPARATOR = '=';

constexpr size_t MINIMUM_VERSION_STRING_SIZE = 3;  // x.y
constexpr size_t MINIMUM_RUNTIME_REQUIREMENTS_SIZE =
    VERSION_KEY.size() + sizeof(KEY_VALUE_SEPARATOR) + MINIMUM_VERSION_STRING_SIZE;  // x.y

void write_requirements_entry(intel_npu::BlobWriterInterface& writer, std::string_view key, std::string_view value) {
    if (!writer.get_offset_relative_to_current_section() == 0) {
        writer.write_from(&REQUIREMENTS_SEPARATOR, sizeof(REQUIREMENTS_SEPARATOR));
    }
    writer.write_from(key.data(), key.size());
    writer.write_from(&KEY_VALUE_SEPARATOR, sizeof(KEY_VALUE_SEPARATOR));
    writer.write_from(value.data(), value.size());
}

}  // namespace

namespace intel_npu {

RuntimeRequirements::RuntimeRequirements(const std::map<SectionID, std::string>& sections_requirements,
                                         const CRE& cre,
                                         const std::unordered_map<SectionID, SectionType>& section_id_to_type)
    : m_sections_requirements(sections_requirements),
      m_cre(cre),
      m_section_id_to_type(section_id_to_type) {}

std::map<SectionID, std::string> RuntimeRequirements::get_sections_requirements() const {
    return m_sections_requirements;
}

CRE RuntimeRequirements::get_cre() const {
    return m_cre;
}

std::unordered_map<SectionID, SectionType> RuntimeRequirements::get_section_id_to_type_mapping() const {
    return m_section_id_to_type;
}

// TODO how to distinguish names
std::unordered_map<SectionID, SectionInstanceEvaluator> RuntimeRequirements::build_section_instance_evaluators(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators) {
    std::unordered_map<SectionID, SectionInstanceEvaluator> per_instance_evaluators;
    // TODO should all instances have evaluators?
    for (const auto [section_id, section_runtime_requirements] : m_sections_requirements) {
        OPENVINO_ASSERT(!per_instance_evaluators.count(section_id),
                        "Found a section that has at least two entries within the runtime requirements");
        OPENVINO_ASSERT(m_section_id_to_type.count(section_id));
        const SectionType section_type = m_section_id_to_type.at(section_id);
        OPENVINO_ASSERT(instance_evaluators.count(section_type),
                        "Missing instance evaluator for section type ",
                        section_type_to_string(section_type));

        per_instance_evaluators[section_id] =
            SectionInstanceEvaluator(instance_evaluators.at(section_type), section_runtime_requirements);
    }

    return per_instance_evaluators;
}

ov::CompatibilityCheck RuntimeRequirements::get_compatibility_check_result(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators) {
    if (!m_compatibility_check_result.has_value()) {
        // TODO maybe log message if caching used, and the new evaluators are ignored
        // TODO asserts evaluators are empty?
        m_type_evaluators = type_evaluators;
        m_instance_evaluators = build_section_instance_evaluators(instance_evaluators);
        m_compatibility_check_result = m_cre.check_compatibility(type_evaluators, m_instance_evaluators);
    }
    return m_compatibility_check_result.value();
}

bool RuntimeRequirements::evaluated() const {
    return m_compatibility_check_result.has_value();
}

std::optional<bool> RuntimeRequirements::get_type_evaluation_result(const SectionType type) const {
    return m_type_evaluators.count(type) && m_type_evaluators.at(type)->evaluated()
               ? std::make_optional<>(m_type_evaluators.at(type)->get_result())
               : std::nullopt;
}

std::optional<bool> RuntimeRequirements::get_instance_evaluation_result(const SectionID id) const {
    return m_instance_evaluators.count(id) && m_instance_evaluators.at(id).evaluated()
               ? std::make_optional<>(m_instance_evaluators.at(id).get_result())
               : std::nullopt;
}

RuntimeRequirementsSection::RuntimeRequirementsSection(const RuntimeRequirements& runtime_requirements,
                                                       const ov::log::Level log_level)
    : ISection(PredefinedSectionType::RUNTIME_REQUIREMENTS),
      m_runtime_requirements(runtime_requirements),
      m_logger("RuntimeRequirementsSection", log_level) {}

void RuntimeRequirementsSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::write");

    // TODO logs?
    // Version
    write_requirements_entry(writer, VERSION_KEY, major_minor_version_to_string(CURRENT_RUNTIME_REQUIREMENTS_VERSION));

    // CRE
    write_requirements_entry(writer, CRE_KEY, cre_to_string(m_runtime_requirements.get_cre()));

    // All section requirements, following the format "<section type name>_<id>=<value>"
    const std::map<SectionID, std::string> sections_requirements = m_runtime_requirements.get_sections_requirements();
    const std::unordered_map<SectionID, SectionType> section_id_to_type =
        m_runtime_requirements.get_section_id_to_type_mapping();

    for (const auto [section_id, section_requirements] : sections_requirements) {
        const SectionType section_type = section_id_to_type.at(section_id);
        write_requirements_entry(writer, section_type_and_id_to_string(section_type, section_id), section_requirements);
    }
}

RuntimeRequirements RuntimeRequirementsSection::get_runtime_requirements() const {
    return m_runtime_requirements;
}

std::shared_ptr<ISection> RuntimeRequirementsSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::read");
    Logger logger("RuntimeRequirementsSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    // TODO test this
    // TODO check manifest section lengths are not greater than the size of the NPU region
    OPENVINO_ASSERT(section_length >= MINIMUM_RUNTIME_REQUIREMENTS_SIZE,
                    "The runtime requirements section is too small");

    // Use the parser
    std::string full_payload(section_length, 0);
    blob_reader.read_into_buffer(full_payload.data(), section_length);

    compat::Parser::attr_map_type parsed_content;
    try {
        compat::Parser parser(full_payload, std::vector<int>());
        parsed_content = parser.getAttributes();
    } catch (const std::exception& ex) {
        OPENVINO_THROW("The content of the runtime requirements section is malformed: ", ex.what());
    }

    // Check the format version
    const MajorMinorVersion parsed_version = major_minor_version_from_string(parsed_content.at(VERSION_KEY.data()));
    OPENVINO_ASSERT(parsed_version == CURRENT_RUNTIME_REQUIREMENTS_VERSION,
                    "Unsupported runtime requirements version: ",
                    parsed_version);
    parsed_content.erase(VERSION_KEY.data());

    // Parse the CRE
    const CRE cre = cre_from_string(parsed_content.at(CRE_KEY.data()));
    parsed_content.erase(CRE_KEY.data());

    // All other entries should have the key format "<section type name>_<id>"
    std::map<SectionID, std::string> sections_requirements;
    std::unordered_map<SectionID, SectionType> section_id_to_type;

    for (const auto [section_type_and_id_string, value] : parsed_content) {
        const auto [section_type, section_id] = section_type_and_id_from_string(section_type_and_id_string);

        OPENVINO_ASSERT(!sections_requirements.count(section_id) && !section_id_to_type.count(section_id),
                        "Found the same section ID more than once within the runtime requirements");
        sections_requirements[section_id] = value;
        section_id_to_type[section_id] = section_type;
    }

    return std::make_shared<RuntimeRequirementsSection>(
        RuntimeRequirements(sections_requirements, cre, section_id_to_type),
        logger.level());
}

bool is_runtime_requirements_format_v3(std::string_view runtime_requirements) {
    try {
        compat::Parser parser(runtime_requirements, std::vector<int>());
        return parser.getAttributes().count(VERSION_KEY.data());
    } catch (const std::exception& ex) {
        OPENVINO_THROW("The content of the runtime requirements section is malformed: ", ex.what());
    }
}

}  // namespace intel_npu
