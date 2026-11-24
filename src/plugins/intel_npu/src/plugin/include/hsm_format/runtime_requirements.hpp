// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../hsm_format/cre.hpp"
#include "../hsm_format/isection_instance_evaluator.hpp"
#include "../hsm_format/isection_type_evaluator.hpp"
#include "../hsm_format/single_section_instance_evaluator.hpp"
#include "hsm_format/isection.hpp"

namespace intel_npu {

/**
 * @brief Class that contains and evaluates the runtime requirements of a compiled model
 */
class RuntimeRequirements {
public:
    /**
     * @param sections_requirements The individual requirements of section instances expressed as strings
     * @param cre The high-level requirements expressed as a logical relationship between section types and/or
     * instances.
     * @param section_id_to_type Mapping from section IDs to types.
     */
    RuntimeRequirements(const std::map<SectionID, std::string>& sections_requirements,
                        const CRE& cre,
                        const std::unordered_map<SectionID, SectionType>& section_id_to_type);

    std::map<SectionID, std::string> get_sections_requirements() const;

    CRE get_cre() const;

    std::unordered_map<SectionID, SectionType> get_section_id_to_type_mapping() const;

    /**
     * @brief Leverage the section type and instance evaluators to evaluate the runtime requirements.
     *
     * @param type_evaluators Instruct the CRE how to evaluate section types. Missing section types will be treated as
     * "supported".
     * @param instance_evaluators Instruct the CRE how to evaluate section instances. Missing section instances will be
     * treated as "supported".
     * @return The final result of the evaluation, along with all types and instances that needed to be evaluated and
     * their evaluation results. The final result can be "supported", "unsupported" or "not applicable". The latter can
     * be returned if the final result is uncertain. This happens through the influence of section instances that
     * evaluate to "not applicable".
     *
     * Only the section types and instances that were required to be evaluated are returned within the remaining
     * attributes. The section types can only evaluate to true or false, while the instance evaluators can return
     * "supported", "unsupported" or "not applicable".
     */
    RuntimeRequirementsEvaluationResult check_compatibility(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
        const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

private:
    /**
     * @brief Consucts the per-instance section evaluators using the all-instances evaluators.
     * @details The all-instance evaluators are the ones received as arguments to the constructor. These instruct how to
     * evaluate the section instances of a given type. The per-instance evaluators leverage the all-instances
     * evaluators, but they also cache the result on top of that.
     */
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> build_single_section_instance_evaluators(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

    /**
     * @brief The individual requirements of section instances expressed as strings
     */
    std::map<SectionID, std::string> m_sections_requirements;
    /**
     * @brief The high-level requirements expressed as a logical relationship between section types and/or
     * instances.
     */
    CRE m_cre;
    std::unordered_map<SectionID, SectionType> m_section_id_to_type;

    Logger m_logger;
};

/**
 * @brief Section able to write and parse the runtime requirements of the whole compiled model
 *
 * @note Overriding "get_cre_tokens" to place the runtime requirements section type would
 * not make sense. Because the plugin importer would be required to understand the runtime requirements to reach that
 * section type.
 */
class RuntimeRequirementsSection final : public ISection {
public:
    RuntimeRequirementsSection(const RuntimeRequirements& runtime_requirements);

    RuntimeRequirements get_runtime_requirements() const;

    /**
     * @brief Writes the formatted compatibility string into the sink.
     * @details The format:
     * version=<major>.<minor>;cre=<cre to string>;<section type name>_<id>=<value>;<section type name>_<id>=<value>;...
     *
     * The "version" entry is mandatory. The "cre" one appears only if the CRE is not empty. The remaining entries are
     * optional, and they correspond to the individual requirements of sections. The order of the entries doesn't
     * matter.
     */
    void encode(const ov::runtime::hsm::SectionSink& sink) const override;

    /**
     * @brief Reconstructs a section instance from a formatted compatibility string.
     * @details The format:
     * version=<major>.<minor>;cre=<cre to string>;<section type name>_<id>=<value>;<section type name>_<id>=<value>;...
     *
     * The "version" entry is mandatory. Only the same major version as the currently used one is supported. All
     * previous minor versions should be supported. The CRE is empty if the "cre" entry is missing. The remaining
     * entries correspond to individual requirements of sections. The order of the entries doesn't matter.
     */
    static ov::runtime::hsm::SectionDecoder<std::shared_ptr<ISection>> decode;

private:
    RuntimeRequirements m_runtime_requirements;

    Logger m_logger;
};

/**
 * @brief Verifies if the version of the given runtime requirements correspond to the blob format V2
 * (header-sections-manifest)
 */
bool is_runtime_requirements_format_v2(std::string_view runtime_requirements);

}  // namespace intel_npu
