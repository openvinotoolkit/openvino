// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection_type_evaluator.hpp"

namespace intel_npu {

ISectionTypeEvaluator::ISectionTypeEvaluator(const CREToken section_type) : m_section_type(section_type) {}

SectionType ISectionTypeEvaluator::get_section_type() const {
    return m_section_type;
}

bool ISectionTypeEvaluator::check_support() const {
    if (m_supported.has_value()) {
        return m_supported.value();
    }

    m_supported = lazy_check_support();
    return m_supported.value();
}

bool ISectionTypeEvaluator::evaluated() const {
    return m_supported.has_value();
}

}  // namespace intel_npu
