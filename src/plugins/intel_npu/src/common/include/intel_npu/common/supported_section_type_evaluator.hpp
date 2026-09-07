// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

static inline const std::unordered_set<SectionType> ALREADY_SUPPORTED_SECTION_TYPES{
    ValidSectionTypeCode::RUNTIME_REQUIREMENTS,
    ValidSectionTypeCode::MANIFEST,
    ValidSectionTypeCode::ELF_MAIN_SCHEDULE,
    ValidSectionTypeCode::ELF_INIT_SCHEDULES,
    ValidSectionTypeCode::DYNAMIC_SCHEDULE,
    ValidSectionTypeCode::IO_LAYOUTS,
    ValidSectionTypeCode::BATCH_SIZE,
    ValidSectionTypeCode::ENCRYPTED_SCHEDULES_FLAG,
    ValidSectionTypeCode::COMPILER_VERSION};

class SupportedSectionTypeEvaluator final : public ISectionTypeEvaluator {
public:
    SupportedSectionTypeEvaluator() = default;

private:
    bool evaluate() const override;
};

}  // namespace intel_npu
