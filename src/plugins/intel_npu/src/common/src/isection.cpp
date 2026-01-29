// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

ISection::ISection(const SectionID section_id) : m_section_id(section_id) {}

SectionID ISection::get_section_id() const {
    return m_section_id;
}

}  // namespace intel_npu
