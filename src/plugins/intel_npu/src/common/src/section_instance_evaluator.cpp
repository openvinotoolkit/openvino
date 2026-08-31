// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/section_instance_evaluator.hpp"

namespace intel_npu {

SectionInstanceEvaluator::SectionInstanceEvaluator(const std::shared_ptr<ISectionInstanceEvaluator>& impl,
                                                   std::string_view runtime_requirements)
    : m_impl(impl),
      m_runtime_requirements(runtime_requirements) {}

bool SectionInstanceEvaluator::get_result() const {
    if (!m_supported.has_value()) {
        m_supported = m_impl->evaluate(m_runtime_requirements);
    }
    return m_supported.value();
}

bool SectionInstanceEvaluator::evaluated() const {
    return m_supported.has_value();
}

}  // namespace intel_npu
