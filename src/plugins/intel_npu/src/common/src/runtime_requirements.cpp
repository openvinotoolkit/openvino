// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

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

bool RuntimeRequirements::get_compatibility_check_result(
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

// TODO "sections_requirements" and CRE as string
void RuntimeRequirementsSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::write");

    writer.write_from(m_cre.get_expression().data(), m_cre.get_expression_length() * sizeof(CREToken));

    m_logger.debug("%lu tokens written", m_cre.get_expression_length());
}

RuntimeRequirements RuntimeRequirementsSection::get_runtime_requirements() const {
    return m_runtime_requirements;
}

std::shared_ptr<ISection> RuntimeRequirementsSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::read");
    Logger logger("RuntimeRequirementsSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length % sizeof(CREToken) == 0,
                    "Received a CRE section length that is not divisible by the CRE token size. Section length: ",
                    section_length,
                    ". CRE token size: ",
                    sizeof(CREToken));
    size_t number_of_tokens = section_length / sizeof(CREToken);
    if (number_of_tokens == 0) {
        logger.warning("The parsed CRE is empty. No compatibility checks will be performed");
    }

    logger.debug("Reading %lu tokens", number_of_tokens);

    std::vector<CREToken> tokens(number_of_tokens);
    blob_reader.read_into_buffer(tokens.data(), number_of_tokens * sizeof(CREToken));

    return std::make_shared<RuntimeRequirementsSection>(CRE(tokens, logger.level()), logger.level());
}

}  // namespace intel_npu
