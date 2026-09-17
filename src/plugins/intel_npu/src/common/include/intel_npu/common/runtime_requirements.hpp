// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cre.hpp"
#include "intel_npu/common/isection.hpp"
#include "isection_instance_evaluator.hpp"
#include "isection_type_evaluator.hpp"
#include "single_section_instance_evaluator.hpp"

namespace intel_npu {

/**
 * @brief Class that contains and evaluates the runtime requirements of a compiled model. The evaluation result is also
 * cached.
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
                        const std::unordered_map<SectionID, SectionType>& section_id_to_type,
                        const ov::log::Level log_level = ov::log::Level::WARNING);

    std::map<SectionID, std::string> get_sections_requirements() const;

    CRE get_cre() const;

    std::unordered_map<SectionID, SectionType> get_section_id_to_type_mapping() const;

    /**
     * @brief Returns the result of evaluating the runtime requirements.
     * @note The result will be cached for future retrieval.
     *
     * @param type_evaluators Instruct the CRE how to evaluate section types. Missing section types will be treated as
     * "unsupported".
     * @param instance_evaluators Instruct the CRE how to evaluate section instances. Missing section instances will be
     * treated as "supported".
     * @return "NOT_APPLICABLE" in uncertain scenarios. "SUPPORTED" or "UNSUPPORTED" when the result is known.
     */
    ov::CompatibilityCheck get_compatibility_check_result(
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& type_evaluators,
        const std::unordered_map<SectionType, std::shared_ptr<ISectionInstanceEvaluator>>& instance_evaluators);

    bool evaluated() const;

    /**
     * @brief Retrieves the result of a section type evaluation.
     * @note This makes sense to be called only after the runtime requirements evaluation has occurred.
     * @return std::nullopt is the section type was not yet evaluated during a CRE evaluation. The evaluation result
     * otherwise.
     */
    std::optional<bool> get_type_evaluation_result(const SectionType type) const;

    /**
     * @brief Retrieves the result of a section instance evaluation.
     * @note This makes sense to be called only after the runtime requirements evaluation has occurred.
     * @return std::nullopt is the section instance was not yet evaluated during a CRE evaluation, or if the result was
     * "NOT_APPLICABLE". The evaluation result otherwise.
     */
    std::optional<bool> get_instance_evaluation_result(const SectionID id) const;

private:
    /**
     * @brief Consucts the per-instance section evaluators using the all-instances evalutors.
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

    /**
     * @brief Instruct the CRE how to evaluate section types. Missing section types will be treated as
     * "unsupported".
     */
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> m_type_evaluators;
    /**
     * @brief Instruct the CRE how to evaluate section instances. Missing section instances will be
     * treated as "supported".
     */
    std::unordered_map<SectionID, SingleSectionInstanceEvaluator> m_instance_evaluators;

    /**
     * @brief Cache for the evaluation result
     */
    std::optional<ov::CompatibilityCheck> m_compatibility_check_result;

    Logger m_logger;
};

/**
 * @brief Section able to write and parse the runtime requirements of the whole compiled model
 */
class RuntimeRequirementsSection final : public ISection {
public:
    RuntimeRequirementsSection(const RuntimeRequirements& runtime_requirements,
                               const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    RuntimeRequirements get_runtime_requirements() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

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
