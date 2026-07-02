// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/supported_section_type_evaluator.hpp"

#include "openvino/core/except.hpp"

namespace intel_npu {

SupportedSectionTypeEvaluator::SupportedSectionTypeEvaluator(const SectionType section_type)
    : ISectionTypeEvaluator(section_type) {}

bool SupportedSectionTypeEvaluator::evaluate() const {
    return true;
}

}  // namespace intel_npu
