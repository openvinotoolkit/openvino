// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include <algorithm>

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

void string_to_lower(std::string& target) {
    std::transform(target.begin(), target.end(), target.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
}

void write_requirements_entry(const ov::runtime::hsm::SectionSink& sink,
                              std::string_view key,
                              std::string_view value,
                              const bool is_first_entry = false) {
    if (!is_first_entry) {
        sink({reinterpret_cast<const std::byte*>(&REQUIREMENTS_SEPARATOR), sizeof(REQUIREMENTS_SEPARATOR)});
    }
    sink({reinterpret_cast<const std::byte*>(key.data()), key.size()});
    sink({reinterpret_cast<const std::byte*>(&KEY_VALUE_SEPARATOR), sizeof(KEY_VALUE_SEPARATOR)});
    sink({reinterpret_cast<const std::byte*>(value.data()), value.size()});
}

}  // namespace

namespace intel_npu {

// TODO use logger
RuntimeRequirements::RuntimeRequirements(const std::map<SectionID, std::string>& sections_requirements,
                                         const CRE& cre,
                                         const std::unordered_map<SectionID, SectionType>& section_id_to_type)
    : m_sections_requirements(sections_requirements),
      m_cre(cre),
      m_section_id_to_type(section_id_to_type),
      m_logger("RuntimeRequirements", Logger::global().level()) {}

std::map<SectionID, std::string> RuntimeRequirements::get_sections_requirements() const {
    return m_sections_requirements;
}

CRE RuntimeRequirements::get_cre() const {
    return m_cre;
}

std::unordered_map<SectionID, SectionType> RuntimeRequirements::get_section_id_to_type_mapping() const {
    return m_section_id_to_type;
}

std::unordered_map<SectionID, SingleSectionInstanceEvaluator>
RuntimeRequirements::build_single_section_instance_evaluators(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators) {
    m_logger.debug("Building the instance evaluators");
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> single_instance_evaluators;

    for (const auto& [section_id, section_runtime_requirements] : m_sections_requirements) {
        OPENVINO_ASSERT(!single_instance_evaluators.count(section_id),
                        "Found a section that has at least two entries within the runtime requirements");
        OPENVINO_ASSERT(m_section_id_to_type.count(section_id));
        const SectionType section_type = m_section_id_to_type.at(section_id);
        OPENVINO_ASSERT(instance_evaluators.count(section_type),
                        "Missing instance evaluator for section type ",
                        section_type.to_string());

        m_logger.trace("Building an instance evaluator for section %s",
                       section_type_and_id_to_string(section_type, section_id).data());

        single_instance_evaluators.emplace(
            section_id,
            SingleSectionInstanceEvaluator(instance_evaluators.at(section_type), section_runtime_requirements));
    }

    return single_instance_evaluators;
}

ov::CompatibilityCheck RuntimeRequirements::get_compatibility_check_result(
    const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
    const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators) {
    // "Unknown" should implicitly evaluate to unsupported. Therefore, no evalutors need to be registered for this type.
    OPENVINO_ASSERT(!type_evaluators.count(SectionTypeCode::UNKNOWN) &&
                    !instance_evaluators.count(SectionTypeCode::UNKNOWN));

    if (!m_compatibility_check_result.has_value()) {
        m_logger.debug("Evaluating the runtime requirements");

        // TODO maybe log message if caching used, and the new evaluators are ignored
        OPENVINO_ASSERT(m_instance_evaluators.empty(), "Invalid state");

        m_type_evaluators = type_evaluators;
        m_instance_evaluators = build_single_section_instance_evaluators(instance_evaluators);
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
    if (!m_instance_evaluators.count(id) || !m_instance_evaluators.at(id).evaluated()) {
        return std::nullopt;
    }

    switch (m_instance_evaluators.at(id).get_result()) {
    case ov::CompatibilityCheck::SUPPORTED:
        return true;
    case ov::CompatibilityCheck::UNSUPPORTED:
        return false;
    case ov::CompatibilityCheck::NOT_APPLICABLE:
        return std::nullopt;
    default:
        OPENVINO_THROW("Unsupported ov::CompatibilityCheck value");
    }
}

RuntimeRequirementsSection::RuntimeRequirementsSection(const RuntimeRequirements& runtime_requirements)
    : ISection(SectionTypeCode::RUNTIME_REQUIREMENTS),
      m_runtime_requirements(runtime_requirements),
      m_logger("RuntimeRequirementsSection", Logger::global().level()) {
    m_logger.debug("Initialized");
}

void RuntimeRequirementsSection::encode(const ov::runtime::hsm::SectionSink& sink) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::write");

    // TODO logs?
    // Version
    write_requirements_entry(sink, VERSION_KEY, CURRENT_RUNTIME_REQUIREMENTS_VERSION.to_string(), true);

    // CRE
    write_requirements_entry(sink, CRE_KEY, m_runtime_requirements.get_cre().to_string());

    // All section requirements, following the format "<section type name>_<id>=<value>"
    const std::map<SectionID, std::string> sections_requirements = m_runtime_requirements.get_sections_requirements();
    const std::unordered_map<SectionID, SectionType> section_id_to_type =
        m_runtime_requirements.get_section_id_to_type_mapping();

    for (const auto& [section_id, section_requirements] : sections_requirements) {
        OPENVINO_ASSERT(section_id_to_type.count(section_id));
        const SectionType section_type = section_id_to_type.at(section_id);
        std::string key = section_type_and_id_to_string(section_type, section_id);
        string_to_lower(key);
        write_requirements_entry(sink, key, section_requirements);
    }
}

RuntimeRequirements RuntimeRequirementsSection::get_runtime_requirements() const {
    return m_runtime_requirements;
}

ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> RuntimeRequirementsSection::decode =
    [](const ov::runtime::hsm::Section& section) -> std::optional<std::shared_ptr<ISection>> {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::read");
    Logger logger("RuntimeRequirementsSection", Logger::global().level());
    OPENVINO_ASSERT(section.tag().id() == static_cast<uint32_t>(SectionTypeCode::RUNTIME_REQUIREMENTS),
                    "Logic error: the decoder received a section of the wrong type");
    OPENVINO_ASSERT(section.tag().is_pointer(), "Logic error: the payload is inline");

    // TODO make sure the inclusion of a runtime requirements section is optional
    const size_t section_length = section.size();
    if (section_length < MINIMUM_RUNTIME_REQUIREMENTS_SIZE) {
        logger.error("The runtime requirements section is too small");
        return std::nullopt;
    }

    // Use the parser
    std::string full_payload(section_length, 0);
    section.read(full_payload.data());

    compat::Parser::attr_map_type parsed_content;
    try {
        compat::Parser parser(full_payload, std::vector<std::string>());
        parsed_content = parser.getAttributes();
    } catch (const std::exception& ex) {
        logger.error("The content of the runtime requirements section is malformed: %s", ex.what());
        return std::nullopt;
    }

    if (!parsed_content.count(VERSION_KEY.data())) {
        logger.error("Missing runtime requirements version");
        return std::nullopt;
    }
    if (!parsed_content.count(CRE_KEY.data())) {
        logger.error("No CRE found within the runtime requirements");
        return std::nullopt;
    }

    // Check the format version
    const MajorMinorVersion parsed_version = MajorMinorVersion::from_string(parsed_content.at(VERSION_KEY.data()));
    if (parsed_version != CURRENT_RUNTIME_REQUIREMENTS_VERSION) {
        logger.error("Unsupported runtime requirements version: %s", parsed_version.to_string());
        return std::nullopt;
    }
    parsed_content.erase(VERSION_KEY.data());

    // Parse the CRE
    logger.debug("Parsing the CRE");
    const CRE cre = CRE::from_string(parsed_content.at(CRE_KEY.data()), logger.level());
    parsed_content.erase(CRE_KEY.data());

    // All other entries should have the key format "<section type name>_<id>"
    logger.debug("Parsing the sections requirements");
    std::map<SectionID, std::string> sections_requirements;
    std::unordered_map<SectionID, SectionType> section_id_to_type;

    for (const auto& [section_type_and_id_string, value] : parsed_content) {
        logger.trace("Parsing the key %s", section_type_and_id_string.data());

        const auto [section_type, section_id] = section_type_and_id_from_string(section_type_and_id_string);
        if (!section_id.has_value()) {
            logger.error("Missing section ID. The keys within the NPU plugin runtime requirements should contain both "
                         "section type and ID");
            return std::nullopt;
        }
        if (sections_requirements.count(section_id.value()) || section_id_to_type.count(section_id.value())) {
            logger.error("Found the same section ID more than once within the runtime requirements");
            return std::nullopt;
        }

        sections_requirements.emplace(section_id.value(), value);
        section_id_to_type.emplace(section_id.value(), section_type);
    }

    // TODO check all classes got their log levels right
    return std::make_shared<RuntimeRequirementsSection>(
        RuntimeRequirements(sections_requirements, cre, section_id_to_type));
};

bool is_runtime_requirements_format_v2(std::string_view runtime_requirements) {
    try {
        compat::Parser parser(runtime_requirements, std::vector<std::string>());
        return parser.getAttributes().count(VERSION_KEY.data());
    } catch (const std::exception& ex) {
        OPENVINO_THROW("The content of the runtime requirements section is malformed: ", ex.what());
    }
}

}  // namespace intel_npu
