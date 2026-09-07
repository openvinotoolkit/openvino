// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

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

class SupportedSectionTypeEvaluator final : public ISectionTypeEvaluator {
public:
    SupportedSectionTypeEvaluator() = default;

private:
    bool evaluate() const override;
};

}  // namespace intel_npu
