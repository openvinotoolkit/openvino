// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cre.hpp"
#include "intel_npu/common/isection.hpp"
#include "isection_instance_evaluator.hpp"
#include "isection_type_evaluator.hpp"
#include "section_instance_evaluator.hpp"

namespace intel_npu {

class RuntimeRequirements {
public:
    RuntimeRequirements(const std::map<SectionID, std::string>& sections_requirements,
                        const CRE& cre,
                        const std::unordered_map<SectionID, SectionType>& section_id_to_type);

    std::map<SectionID, std::string> get_sections_requirements() const;

    CRE get_cre() const;

    std::unordered_map<SectionID, SectionType> get_section_id_to_type_mapping() const;

    bool get_compatibility_check_result(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
        const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

private:
    std::unordered_map<SectionID, SectionInstanceEvaluator> build_section_instance_evaluators(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

    std::map<SectionID, std::string> m_sections_requirements;
    CRE m_cre;
    std::unordered_map<SectionID, SectionType> m_section_id_to_type;

    std::optional<bool> m_compatibility_check_result;
};

/**
 * @brief Section able to write the runtime requirements of the whole model
 */
class RuntimeRequirementsSection final : public ISection {
public:
    /**
     * @brief Construct a section able to write the runtime requirements of the whole model.
     *
     * @param sections_requirements A mapping from <string section type>_<section ID> to the corresponding runtime
     * requirements. The map is ordered, in order to assure subsequent writes produce identical blobs.
     * @param cre The compatiblity requirements expression, tying together the requirements of individual sections.
     */
    RuntimeRequirementsSection(const RuntimeRequirements& runtime_requirements,
                               const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    RuntimeRequirements get_runtime_requirements() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    RuntimeRequirements m_runtime_requirements;

    Logger m_logger;
};

}  // namespace intel_npu
