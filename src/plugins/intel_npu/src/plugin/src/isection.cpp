// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "isection.hpp"

namespace intel_npu {

ISection::ISection(const SectionID section_id) : m_section_id(section_id) {}

ISection::SectionID ISection::get_section_id() {
    return m_section_id;
}

std::optional<uint64_t> ISection::get_length() {
    return std::nullopt;
}

}  // namespace intel_npu
