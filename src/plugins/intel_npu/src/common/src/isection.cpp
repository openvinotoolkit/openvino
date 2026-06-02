// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

ISection::ISection(const SectionType type) : m_section_type(type) {}

SectionType ISection::get_section_type() const {
    return m_section_type;
}

std::optional<SectionTypeInstance> ISection::get_section_type_instance() const {
    return m_section_type_instance;
}

void ISection::set_section_type_instance(const SectionTypeInstance type_instance) const {
    OPENVINO_ASSERT(!m_section_type_instance.has_value(),
                    "Attempted to set an instance ID to a section that already had one. Section type: ",
                    m_section_type,
                    ", old instance ID: ",
                    m_section_type_instance.value());

    m_section_type_instance = type_instance;
}

std::optional<SectionID> ISection::get_section_id() const {
    if (!m_section_type_instance.has_value()) {
        return std::nullopt;
    }

    return SectionID(m_section_type, m_section_type_instance.value());
}

SectionID::SectionID(SectionType section_type, SectionTypeInstance section_type_instance) {
    type = section_type;
    type_instance = section_type_instance;
}

bool operator==(const SectionID& sid1, const SectionID& sid2) {
    return sid1.type == sid2.type && sid1.type_instance == sid2.type_instance;
}

std::ostream& operator<<(std::ostream& out, const SectionID& id) {
    out << "(type=" << id.type << ", instance=" << id.type_instance << ")";
    return out;
}

}  // namespace intel_npu

size_t std::hash<intel_npu::SectionID>::operator()(const intel_npu::SectionID& sid) const {
    return std::hash<intel_npu::SectionType>{}(sid.type) ^
           (std::hash<intel_npu::SectionTypeInstance>{}(sid.type_instance) << 1);
}
