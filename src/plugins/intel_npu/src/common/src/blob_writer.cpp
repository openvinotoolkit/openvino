// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_writer.hpp"

#include <iterator>

#include "intel_npu/common/blob_reader.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

constexpr intel_npu::SectionTypeInstance FIRST_INSTANCE_ID = 0;

}  // namespace

namespace intel_npu {

BlobWriterInterface::BlobWriterInterface(
    std::ostream& stream,
    const std::queue<std::shared_ptr<ISection>>& registered_sections,
    const CRE& cre,
    const std::unordered_map<SectionType, SectionTypeInstance>& next_type_instance_id)
    : m_stream(stream),
      m_next_type_instance_id(next_type_instance_id),
      m_registered_sections(registered_sections),
      m_cre(cre),
      m_stream_npu_region_start(stream.tellp()),
      m_logger("BlobWriterInterface", Logger::global().level()) {}

SectionTypeInstance BlobWriterInterface::register_section(const std::shared_ptr<ISection>& section) {
    const SectionType section_type = section->get_section_type();
    if (!m_next_type_instance_id.count(section_type)) {
        m_next_type_instance_id[section_type] = FIRST_INSTANCE_ID;
    }

    const SectionTypeInstance type_instance_id = m_next_type_instance_id[section_type]++;
    section->set_section_type_instance(type_instance_id);
    m_registered_sections.push(section);

    return type_instance_id;
}

void BlobWriterInterface::append_compatibility_requirement(const CRE::Token requirement_token) {
    m_cre.append_to_expression(requirement_token);
}

void BlobWriterInterface::append_compatibility_requirement(const std::vector<CRE::Token>& requirement_tokens) {
    m_cre.append_to_expression(requirement_tokens);
}

void BlobWriterInterface::write(const void* source, const size_t size) {
    OPENVINO_ASSERT(m_stream.get().good());
    m_stream.get().write(reinterpret_cast<const char*>(source), size);
}

void BlobWriterInterface::add_padding(const size_t size) {
    if (size > 0) {
        std::fill_n(std::ostream_iterator<char>(m_stream.get()), size, 0);
    }
}

std::streamoff BlobWriterInterface::get_offset_relative_to_current_section() const {
    OPENVINO_ASSERT(m_stream.get().good());
    return m_stream.get().tellp() - m_stream_current_section_start;
}

void BlobWriterInterface::move_cursor_relative_to_current_section(const size_t offset) {
    OPENVINO_ASSERT(m_stream.get().good());
    m_stream.get().seekp(m_stream_npu_region_start + static_cast<std::streamoff>(offset));
}

std::streamoff BlobWriterInterface::get_offset_relative_to_npu_region() const {
    OPENVINO_ASSERT(m_stream.get().good());
    return m_stream.get().tellp() - m_stream_npu_region_start;
}

void BlobWriterInterface::move_cursor_relative_to_npu_region(const size_t offset) {
    OPENVINO_ASSERT(m_stream.get().good());
    OPENVINO_ASSERT(m_stream_current_section_start >= m_stream_npu_region_start);
    OPENVINO_ASSERT(offset >= static_cast<size_t>(m_stream_current_section_start - m_stream_npu_region_start),
                    "A section writer has attempted a jump outside the boundaries of its own payload. Jump location: ",
                    offset,
                    ". Minimum allowed value: ",
                    m_stream_current_section_start - m_stream_npu_region_start);
    m_stream.get().seekp(m_stream_npu_region_start + static_cast<std::streamoff>(offset));
}

void BlobWriterInterface::seek_to_the_end() {
    OPENVINO_ASSERT(m_stream.get().good());
    m_stream.get().seekp(0, std::ios_base::end);
}

BlobWriter::BlobWriter() : m_logger("BlobWriter", Logger::global().level()) {
    append_compatibility_requirement(CRE::PredefinedCapabilityToken::CRE_EVALUATION);
}

BlobWriter::BlobWriter(const std::shared_ptr<BlobReader>& blob_reader)
    : m_logger("BlobWriter", Logger::global().level()) {
    // TODO review the class const qualifiers
    const auto cre_section = blob_reader->retrieve_section(CRE_SECTION_ID);
    OPENVINO_ASSERT(cre_section != nullptr, "The CRE section was not found within the BlobReader");

    m_cre = std::dynamic_pointer_cast<CRESection>(cre_section)->get_cre();

    for (const auto& [section_type_id, sections] : blob_reader->m_parsed_sections) {
        // The CRE & offsets table sections are added by the write() method after writing all registered sections (jic
        // the registered sections will alter the CRE/table). Therefore, these sections should be omitted here.
        if (section_type_id != PredefinedSectionType::OFFSETS_TABLE && section_type_id != PredefinedSectionType::CRE) {
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

void BlobWriter::write_section(const std::unique_ptr<BlobWriterInterface>& blob_writer_interface,
                               const std::shared_ptr<ISection>& section,
                               OffsetsTable& offsets_table) {
    blob_writer_interface->seek_to_the_end();
    blob_writer_interface->m_stream_current_section_start = blob_writer_interface->m_stream.get().tellp();
    const uint64_t offset = blob_writer_interface->get_offset_relative_to_npu_region();

    section->write(blob_writer_interface);

    blob_writer_interface->seek_to_the_end();
    const uint64_t length = static_cast<uint64_t>(blob_writer_interface->get_offset_relative_to_npu_region() - offset);

    // All sections registered within the BlobWriter are automatically added to the table of offsets
    const std::optional<SectionID> section_id = section->get_section_id();
    // The instance ID should have been added by the writer. Therefore, the section ID should exist.
    OPENVINO_ASSERT(section_id.has_value(), "Missing section ID while writing the section");
    offsets_table.add_entry(section_id.value(), offset, length);
}

void BlobWriter::write(std::ostream& stream) {
    // Only the attributes within this object will be altered. This is done to ensure write idempotency and thread
    // safety
    auto blob_writer_interface =
        std::make_unique<BlobWriterInterface>(stream, m_registered_sections, m_cre, m_next_type_instance_id);

    // The table of offsets corresponds to a single blob written into a stream. Therefore, this table should exist
    // only within the scope of the writing session.
    OffsetsTable offsets_table;

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

        write_section(blob_writer_interface, section, offsets_table);
    }

    // Write the CRESection
    // Note: this was left near the end jic some writers had to register some more capability IDs for some reason
    // TODO: in that case, reading this blob and then writing it again would add redundant CRE tokens. Maybe a redesign
    // would be useful here.
    const auto cre_section = std::make_shared<CRESection>(m_cre);
    cre_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(blob_writer_interface, cre_section, offsets_table);

    // Write the table of offsets
    offsets_table_location = blob_writer_interface->get_offset_relative_to_npu_region();

    const auto offsets_table_section = std::make_shared<OffsetsTableSection>(offsets_table);
    offsets_table_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(blob_writer_interface, offsets_table_section, offsets_table);

    npu_region_size = blob_writer_interface->get_offset_relative_to_npu_region();
    offsets_table_size = npu_region_size - offsets_table_location;

    // Go back to the beginning and write the size of the whole NPU region & the location of the offsets table
    stream.seekp(will_come_back_to_this_at_the_end);
    stream.write(reinterpret_cast<const char*>(&npu_region_size), sizeof(npu_region_size));
    stream.write(reinterpret_cast<const char*>(&offsets_table_location), sizeof(offsets_table_location));
    stream.write(reinterpret_cast<const char*>(&offsets_table_size), sizeof(offsets_table_size));
}

}  // namespace intel_npu
