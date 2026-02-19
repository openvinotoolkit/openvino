// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_writer.hpp"

#include "intel_npu/common/blob_reader.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

constexpr intel_npu::SectionTypeInstance FIRST_INSTANCE_ID = 0;

}  // namespace

namespace intel_npu {

BlobWriter::BlobWriter() : m_logger("BlobWriter", Logger::global().level()) {
    append_compatibility_requirement(CRE::PredefinedCapabilityToken::CRE_EVALUATION);
}

BlobWriter::BlobWriter(const std::shared_ptr<BlobReader>& blob_reader)
    : m_offsets_table(blob_reader->m_offsets_table),
      m_logger("BlobWriter", Logger::global().level()) {
    // TODO review the class const qualifiers
    const auto cre_section = blob_reader->retrieve_section(CRE_SECTION_ID);
    OPENVINO_ASSERT(cre_section != nullptr, "The CRE section was not found within the BlobReader");

    m_cre = std::dynamic_pointer_cast<CRESection>(cre_section)->get_cre();

    for (const auto& [section_type_id, sections] : blob_reader->m_parsed_sections) {
        // The offsets table section is added by the write() method after writing all registered sections (jic the
        // registered sections will alter the table). Therefore, this section should be omitted here.
        if (section_type_id != PredefinedSectionType::OFFSETS_TABLE) {
            // Recall that each section type can have multiple instances
            for (const auto& [instance_id, section] : sections) {
                register_section_from_blob_reader(section);
            }
        }
    }
}

SectionTypeInstance BlobWriter::register_section(const std::shared_ptr<ISection>& section) {
    const SectionType section_type = section->get_section_type();
    if (!m_next_type_instance_id.count(section_type)) {
        m_next_type_instance_id[section_type] = FIRST_INSTANCE_ID;
    }

    const SectionTypeInstance type_instance_id = m_next_type_instance_id[section_type]++;
    section->set_section_type_instance(type_instance_id);
    m_registered_sections.push(section);

    return type_instance_id;
}

void BlobWriter::register_section_from_blob_reader(const std::shared_ptr<ISection>& section) {
    const SectionType section_type = section->get_section_type();
    if (!m_next_type_instance_id.count(section_type)) {
        m_next_type_instance_id[section_type] = FIRST_INSTANCE_ID;
    }

    // Update the next instance ID to be used.
    // Note: not sure if we really need to do this, since supposedly there won't be any other sections registered by the
    // plugin in this case. A blob that was imported should already contain all the sections it needs.
    OPENVINO_ASSERT(section->get_section_type_instance().has_value());
    const SectionTypeInstance candidate = section->get_section_type_instance().value() + 1;
    m_next_type_instance_id[section_type] =
        candidate > m_next_type_instance_id[section_type] ? candidate : m_next_type_instance_id[section_type];

    m_registered_sections.push(section);
}

void BlobWriter::append_compatibility_requirement(const CRE::Token requirement_token) {
    m_cre.append_to_expression(requirement_token);
}

void BlobWriter::append_compatibility_requirement(const std::vector<CRE::Token>& requirement_tokens) {
    m_cre.append_to_expression(requirement_tokens);
}

std::streamoff BlobWriter::get_stream_relative_position(std::ostream& stream) const {
    OPENVINO_ASSERT(m_stream_base.has_value());
    OPENVINO_ASSERT(stream.good());
    return stream.tellp() - m_stream_base.value();
}

void BlobWriter::move_stream_cursor_to_relative_position(std::ostream& stream,
                                                         const SectionID section_id,
                                                         const uint64_t offset) {
    OPENVINO_ASSERT(m_stream_base.has_value());
    OPENVINO_ASSERT(stream.good());  // TODO maybe the stream should be an attribute exposed via the BlobWriter's API

    // Bound checking. Sections should jump only to locations within their payload.
    const std::optional<uint64_t> section_offset = m_offsets_table.lookup_offset(section_id);
    OPENVINO_ASSERT(section_offset.has_value(), "Section ID not found within the table of offsets: ", section_id);
    const std::optional<uint64_t> section_length = m_offsets_table.lookup_length(section_id);

    OPENVINO_ASSERT(offset >= section_offset.value() && offset < section_offset.value() + section_length.value(),
                    "Section using the Section ID ",
                    section_id,
                    " attempted a jump outside the boundaries of its own payload. Jump location: ",
                    offset,
                    ". Boundaries: [",
                    section_offset.value(),
                    ", ",
                    section_offset.value() + section_length.value(),
                    "].");

    stream.seekp(m_stream_base.value() + static_cast<std::streamoff>(offset));
}

void BlobWriter::write_section(std::ostream& stream, const std::shared_ptr<ISection>& section) {
    stream.seekp(0, std::ios_base::end);
    const uint64_t offset = get_stream_relative_position(stream);
    auto position_before_write = stream.tellp();

    section->write(stream, this);

    stream.seekp(0, std::ios_base::end);
    const uint64_t length = static_cast<uint64_t>(stream.tellp() - position_before_write);

    // All sections registered within the BlobWriter are automatically added to the table of offsets
    const std::optional<SectionID> section_id = section->get_section_id();
    // The instance ID should have been added by the writer. Therefore, the section ID should exist.
    OPENVINO_ASSERT(section_id.has_value(), "Missing section ID while writing the section");
    m_offsets_table.add_entry(section_id.value(), offset, length);
}

void BlobWriter::write(std::ostream& stream) {
    // Backup the attributes of the class. Writing to a stream needs to be idempotent
    std::queue<std::shared_ptr<ISection>> registered_sections_backup(m_registered_sections);
    CRE cre_clone = m_cre;
    OffsetsTable offsets_table_backup = m_offsets_table;

    // The NPU specific region starts from here
    m_stream_base = stream.tellp();

    // The region of persistent format (fields of cemented location and meaning)
    stream.write(reinterpret_cast<const char*>(MAGIC_BYTES.data()), MAGIC_BYTES.size());
    stream.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

    // Stop condition for the BlobReader: the size of the data written here
    const auto will_come_back_to_this_at_the_end = stream.tellp();
    uint64_t npu_region_size = 0;
    stream.write(reinterpret_cast<const char*>(&npu_region_size),
                 sizeof(npu_region_size));  // placeholder

    // Placeholder until the offsets table is fully populated and written into the blob
    uint64_t offsets_table_location = 0;
    uint64_t offsets_table_size = 0;
    stream.write(reinterpret_cast<const char*>(&offsets_table_location), sizeof(offsets_table_location));
    stream.write(reinterpret_cast<const char*>(&offsets_table_size), sizeof(offsets_table_size));

    // The region of non-persistent format (list of key-length-payload sections, any order & no restrictions w.r.t. the
    // content of the payload)
    while (!m_registered_sections.empty()) {
        const std::shared_ptr<ISection>& section = m_registered_sections.front();
        m_registered_sections.pop();

        write_section(stream, section);
    }

    // Write the CRESection
    // Note: this was left near the end jic some writers had to register some more capability IDs for some reason
    // TODO: in that case, reading this blob and then writing it again would add redundant CRE tokens. Maybe a redesign
    // would be useful here.
    const auto cre_section = std::make_shared<CRESection>(m_cre);
    cre_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(stream, cre_section);

    // Write the table of offsets
    offsets_table_location = get_stream_relative_position(stream);

    const auto offsets_table_section = std::make_shared<OffsetsTableSection>(m_offsets_table);
    offsets_table_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(stream, offsets_table_section);

    offsets_table_size = get_stream_relative_position(stream) - offsets_table_location;
    npu_region_size = get_stream_relative_position(stream);

    // Go back to the beginning and write the size of the whole NPU region & the location of the offsets table
    stream.seekp(will_come_back_to_this_at_the_end);
    stream.write(reinterpret_cast<const char*>(&npu_region_size), sizeof(npu_region_size));
    stream.write(reinterpret_cast<const char*>(&offsets_table_location), sizeof(offsets_table_location));
    stream.write(reinterpret_cast<const char*>(&offsets_table_size), sizeof(offsets_table_size));

    // Restore the attributes
    m_registered_sections = std::move(registered_sections_backup);
    m_offsets_table = std::move(offsets_table_backup);
    m_cre = std::move(cre_clone);
}

}  // namespace intel_npu
