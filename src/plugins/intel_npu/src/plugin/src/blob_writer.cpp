// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_writer.hpp"

namespace intel_npu {

BlobWriter::BlobWriter() : m_cre(std::make_shared<CRESection>()) {
    register_section(m_cre);
}

void BlobWriter::register_section(const std::shared_ptr<ISection>& section) {
    m_registered_sections.push_back(section);
    OPENVINO_ASSERT(!m_registered_sections_ids.count(section->get_section_id()));
    m_registered_sections_ids.insert(section->get_section_id());
}

void BlobWriter::append_compatibility_requirement(const CREToken requirement_token) {
    m_cre->append_to_expression(requirement_token);
}

void BlobWriter::register_offset_in_table(const ISection::SectionID id, const uint64_t offset) {
    OPENVINO_ASSERT(!m_offsets_table.count(id));
    m_offsets_table[id] = offset;
}

}  // namespace intel_npu
