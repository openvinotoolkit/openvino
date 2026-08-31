// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/runtime_requirements.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

RuntimeRequirements::RuntimeRequirements(const CRE& cre, const std::map<SectionID, std::string>& sections_requirements)
    : m_cre(cre),
      m_sections_requirements(sections_requirements) {}

// TODO how to distinguish names
std::unordered_map<SectionID, SectionInstanceEvaluator> RuntimeRequirements::build_section_instance_evaluators(
    const std::unordered_map<SectionID, ISectionInstanceEvaluator>& instance_evaluators) {
    std::unordered_map<SectionID, SectionInstanceEvaluator> per_instance_evaluators;
    // TODO should all instances have evaluators?
}

bool RuntimeRequirements::get_compatibility_check_result(
    const std::unordered_map<SectionType, ISectionTypeEvaluator>& type_evaluators,
    const std::unordered_map<SectionID, ISectionInstanceEvaluator>& instance_evaluators) {
    if (!m_compatibility_check_result.has_value()) {
    }
    return m_compatibility_check_result.value();
}

RuntimeRequirementsSection::RuntimeRequirementsSection(const std::map<SectionID, std::string>& sections_requirements,
                                                       const CRE& cre,
                                                       const ov::log::Level log_level)
    : ISection(PredefinedSectionType::RUNTIME_REQUIREMENTS),
      m_sections_requirements(sections_requirements),
      m_cre(cre),
      m_logger("RuntimeRequirementsSection", log_level) {}

// TODO "sections_requirements" and CRE as string
void RuntimeRequirementsSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "RuntimeRequirementsSection::write");

    writer.write_from(m_cre.get_expression().data(), m_cre.get_expression_length() * sizeof(CREToken));

    m_logger.debug("%lu tokens written", m_cre.get_expression_length());
}

CRE RuntimeRequirementsSection::get_cre() const {
    return m_cre;
}

std::map<SectionID, std::string> RuntimeRequirementsSection::get_sections_requirements() const {
    return m_sections_requirements;
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
