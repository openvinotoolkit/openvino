// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection_instance_evaluator.hpp"

namespace intel_npu {

ISectionInstanceEvaluator::ISectionInstanceEvaluator(const CREToken section_type) : m_section_type(section_type) {}

SectionType ISectionInstanceEvaluator::get_section_type() const {
    return m_section_type;
}

}  // namespace intel_npu
