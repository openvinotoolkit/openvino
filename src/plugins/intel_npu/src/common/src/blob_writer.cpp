// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_writer.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/offsets_table.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

}  // namespace

namespace intel_npu {

BlobWriter::BlobWriter()
    : m_cre(std::make_shared<CRESection>()),
      m_offsets_table(std::make_shared<std::unordered_map<SectionID, uint64_t>>()),
      m_logger("BlobWriter", Logger::global().level()) {
    append_compatibility_requirement(CRE::PredefinedCapabilityToken::CRE_EVALUATION);
}

BlobWriter::BlobWriter(const std::shared_ptr<BlobReader>& blob_reader)
    : m_cre(std::dynamic_pointer_cast<CRESection>(blob_reader->retrieve_section(PredefinedSectionID::CRE))),
      m_offsets_table(blob_reader->m_offsets_table),
      m_logger("BlobWriter", Logger::global().level()) {
    // TODO review the class const qualifiers
    std::unordered_map<SectionID, std::shared_ptr<ISection>> m_parsed_sections;
    for (const auto& [section_id, section] : m_parsed_sections) {
        // The offsets table section is added by the write() method after writing all registered sections (jic the
        // registered sections will alter the table). Therefore, this section should be omitted here.
        if (section_id != PredefinedSectionID::OFFSETS_TABLE) {
            register_section(section);
        }
    }
}

void BlobWriter::register_section(const std::shared_ptr<ISection>& section) {
    m_registered_sections.push(section);
    OPENVINO_ASSERT(!m_registered_sections_ids.count(section->get_section_id()));
    m_registered_sections_ids.insert(section->get_section_id());
}

void BlobWriter::append_compatibility_requirement(const CRE::Token requirement_token) {
    m_cre->append_to_expression(requirement_token);
}

void BlobWriter::register_offset_in_table(const SectionID id, const uint64_t offset) {
    OPENVINO_ASSERT(!m_offsets_table->count(id));
    (*m_offsets_table)[id] = offset;
}

std::streamoff BlobWriter::get_stream_relative_position(std::ostream& stream) const {
    OPENVINO_ASSERT(m_stream_base.has_value());
    OPENVINO_ASSERT(stream.good());
    return stream.tellp() - m_stream_base.value();
}

void BlobWriter::write_section(std::ostream& stream, const std::shared_ptr<ISection>& section) {
    const SectionID section_id = section->get_section_id();

    // All sections registered within the BlobWriter are automatically added to the table of offsets
    register_offset_in_table(section_id, get_stream_relative_position(stream));

    stream.write(reinterpret_cast<const char*>(&section_id), sizeof(section_id));

    std::optional<uint64_t> length = section->get_length();

    if (length.has_value()) {
        stream.write(reinterpret_cast<const char*>(&length.value()), sizeof(length.value()));
        auto position_before_write = stream.tellp();
        section->write(stream, this);

        OPENVINO_ASSERT(length.value() == static_cast<uint64_t>(stream.tellp() - position_before_write),
                        "Mismatch between the length provided by the section class and the size written in the "
                        "blob. Section ID: ",
                        section_id);
    } else {
        // Use the cursor to deduce the length
        uint64_t length = 0;  // placeholder
        auto length_location = stream.tellp();
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));

        const auto payload_start = stream.tellp();
        section->write(stream, this);
        stream.seekp(0, std::ios_base::end);

        // Compute the size of the payload and then go back and write the true value
        length = stream.tellp() - payload_start;
        stream.seekp(length_location);
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
    }

    stream.seekp(0, std::ios_base::end);
}

void BlobWriter::write(std::ostream& stream) {
    // Backup the attributes of the class. Writing to a stream needs to be idempotent
    std::queue<std::shared_ptr<ISection>> registered_sections_backup(m_registered_sections);
    std::shared_ptr<CRESection> cre_clone = m_cre->clone();
    auto offsets_table_backup = std::make_shared<std::unordered_map<SectionID, uint64_t>>(*m_offsets_table);

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
    stream.write(reinterpret_cast<const char*>(&offsets_table_location), sizeof(offsets_table_location));

    // The region of non-persistent format (list of key-length-payload sections, any order & no restrictions w.r.t. the
    // content of the payload)
    while (!m_registered_sections.empty()) {
        const std::shared_ptr<ISection>& section = m_registered_sections.front();
        m_registered_sections.pop();

        write_section(stream, section);
    }

    // Write the CRESection
    // Note: this was left near the end jic some writers had to register some more capability IDs for some reason
    write_section(stream, m_cre);

    // TODO should the CRESection also be left at the end, jic another writer still has to append to it?
    // Write the table of offsets
    offsets_table_location = get_stream_relative_position(stream);

    const auto offsets_table_section = std::make_shared<OffsetsTableSection>(m_offsets_table);
    write_section(stream, offsets_table_section);

    npu_region_size = get_stream_relative_position(stream);

    // Go back to the beginning and write the size of the whole NPU region & the location of the offsets table
    stream.seekp(will_come_back_to_this_at_the_end);
    stream.write(reinterpret_cast<const char*>(&npu_region_size), sizeof(npu_region_size));
    stream.write(reinterpret_cast<const char*>(&offsets_table_location), sizeof(offsets_table_location));

    // Restore the attributes
    m_registered_sections = registered_sections_backup;
    m_offsets_table = offsets_table_backup;
    m_cre = cre_clone;
}

}  // namespace intel_npu
