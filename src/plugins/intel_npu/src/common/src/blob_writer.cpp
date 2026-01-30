// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_writer.hpp"

#include "intel_npu/common/blob_reader.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

}  // namespace

namespace intel_npu {

BlobWriter::BlobWriter() : m_logger("BlobWriter", Logger::global().level()) {
    append_compatibility_requirement(CRE::PredefinedCapabilityToken::CRE_EVALUATION);
}

BlobWriter::BlobWriter(const std::shared_ptr<BlobReader>& blob_reader)
    : m_cre(std::dynamic_pointer_cast<CRESection>(blob_reader->retrieve_section(PredefinedSectionID::CRE))->get_cre()),
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

void BlobWriter::write_section(std::ostream& stream, const std::shared_ptr<ISection>& section) {
    const SectionID section_id = section->get_section_id();

    stream.seekp(0, std::ios_base::end);
    const uint64_t offset = get_stream_relative_position(stream);
    auto position_before_write = stream.tellp();

    section->write(stream, this);

    stream.seekp(0, std::ios_base::end);
    const uint64_t length = static_cast<uint64_t>(stream.tellp() - position_before_write);

    // All sections registered within the BlobWriter are automatically added to the table of offsets
    m_offsets_table.add_entry(section_id, offset, length);
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
    // TODO: in that case, read then write would fill a bit of junk
    const auto cre_section = std::make_shared<CRESection>(m_cre);
    write_section(stream, cre_section);

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
    m_registered_sections = std::move(registered_sections_backup);
    m_offsets_table = std::move(offsets_table_backup);
    m_cre = std::move(cre_clone);
}

}  // namespace intel_npu
