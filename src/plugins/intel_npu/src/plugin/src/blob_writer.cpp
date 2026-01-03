// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_writer.hpp"

namespace {

// TODO: find out why these are like this
constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

}  // namespace

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

void BlobWriter::write(std::ostream& stream) {
    stream_base = stream.tellp();
    cursor = 0;

    // The region of persistent format (fields of cemented location and meaning)
    stream.write(reinterpret_cast<const char*>(MAGIC_BYTES.data()), MAGIC_BYTES.size());
    cursor += MAGIC_BYTES.size();  // TODO more elegant pls
    stream.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));
    cursor += sizeof(FORMAT_VERSION);
    const uint64_t will_come_back_to_this_at_the_end = cursor;
    stream.write(reinterpret_cast<const char*>(&offsets_table_location),
                 sizeof(offsets_table_location));  // placeholder
    cursor += sizeof(offsets_table_location);

    // The region of non-persistent format (list of key-length-payload sections, any order & no restrictions w.r.t. the
    // content of the payload)
    for (const std::shared_ptr<ISection>& section : m_registered_sections) {
        const ISection::SectionID section_id = section->get_section_id();
        stream.write(reinterpret_cast<const char*>(&section_id), sizeof(section_id));
        cursor += sizeof(section_id);

        uint64_t length = 0;  // placeholder
        auto length_location = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        cursor += sizeof(section_id);

        const uint64_t payload_start = cursor;
        section->write(stream, this);

        // Compute the size of the payload and then go back and write the true value
        length = cursor - payload_start;
        stream.seekp(length_location);
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        stream.seekp(stream_base + cursor);
    }

    // TODO: define the offsets table section. then here you should retrieve its location and write it at the beg
}

}  // namespace intel_npu
