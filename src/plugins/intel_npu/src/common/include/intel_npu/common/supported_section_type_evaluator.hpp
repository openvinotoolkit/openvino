// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

/**
 * @brief These section types are supported by the current version of the plugin. No addditional checks required.
 */
static inline const std::unordered_set<SectionType> ALREADY_SUPPORTED_SECTION_TYPES{
    SectionTypeCode::RUNTIME_REQUIREMENTS,
    SectionTypeCode::MANIFEST,
    SectionTypeCode::ELF_MAIN_SCHEDULE,
    SectionTypeCode::ELF_INIT_SCHEDULES,
    SectionTypeCode::DYNAMIC_SCHEDULE,
    SectionTypeCode::IO_LAYOUTS,
    SectionTypeCode::BATCH_SIZE,
    SectionTypeCode::ENCRYPTED_SCHEDULES_FLAG,
    SectionTypeCode::COMPILER_VERSION};

/**
 * @brief Singleton class that can be associated to all section types that are already supported by the plugin (i.e. no
 * additional support checks required).
 * @details Always evaluates to "true"
 */
class SupportedSectionTypeEvaluator : public ISectionTypeEvaluator {
public:
    SupportedSectionTypeEvaluator(const SupportedSectionTypeEvaluator&) = delete;

    SupportedSectionTypeEvaluator(SupportedSectionTypeEvaluator&&) = delete;

    SupportedSectionTypeEvaluator& operator=(const SupportedSectionTypeEvaluator&) = delete;

    SupportedSectionTypeEvaluator& operator=(SupportedSectionTypeEvaluator&&) = delete;

    static std::shared_ptr<SupportedSectionTypeEvaluator> get_instance();

protected:
    SupportedSectionTypeEvaluator() = default;

private:
    bool evaluate() const override;
};

}  // namespace intel_npu
