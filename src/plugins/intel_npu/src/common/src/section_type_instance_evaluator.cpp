// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_type_instance_evaluator.hpp"

namespace intel_npu {

SectionTypeInstanceEvaluator::SectionTypeInstanceEvaluator(const std::shared_ptr<ISection>& section,
                                                           BlobReaderInterface reader)
    : m_section(section),
      m_reader(std::move(reader)) {}

bool SectionTypeInstanceEvaluator::check_support() {
    if (m_supported.has_value()) {
        return m_supported.value();
    }

    m_supported = m_section->evaluate_compatibility_based_on_section_content(m_reader);
    return m_supported.value();
}

}  // namespace intel_npu
