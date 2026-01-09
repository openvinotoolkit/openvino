// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader.hpp"

#include "intel_npu/common/offsets_table.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

}  // namespace

namespace intel_npu {

BlobReader::BlobReader(std::istream& source) : m_source(source), m_stream_base(source.tellg()) {}

BlobReader::BlobReader(const ov::Tensor& source) : m_source(source), m_cursor(0) {}

void BlobReader::register_reader(
    const SectionID section_id,
    std::function<std::shared_ptr<ISection>(const BlobSource&, const std::unordered_map<SectionID, uint64_t>&)>
        reader) {
    m_readers[section_id] = reader;
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID section_id) {
    auto search_result = m_parsed_sections.find(section_id);
    if (search_result != m_parsed_sections.end()) {
        return search_result->second;
    }
    return nullptr;
}

// TODO test the windows debug build works properly if using the "better" implementation
// TODO allow reinterpreting instead of copying
void BlobReader::read_data_from_source(char* destination, const size_t size) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        stream->get().read(destination, size);
    } else if (const std::reference_wrapper<const ov::Tensor>* tensor =
                   std::get_if<std::reference_wrapper<const ov::Tensor>>(&m_source)) {
        std::memcpy(destination, tensor->get().data<const char>() + m_cursor, size);
        m_cursor += size;
    }
}

size_t BlobReader::get_cursor_position() {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        return stream->get().tellg() - m_stream_base;
    }
    return m_cursor;
}

void BlobReader::move_cursor(const size_t offset) {
    if (const std::reference_wrapper<std::istream>* stream =
            std::get_if<std::reference_wrapper<std::istream>>(&m_source)) {
        stream->get().seekg(static_cast<size_t>(m_stream_base) + offset);
    }
    m_cursor = offset;
}

void BlobReader::read(const std::unordered_set<CRE::Token>& plugin_capabilities_ids) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    read_data_from_source(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    read_data_from_source(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    // Step 1: Read the table of offsets
    SectionID section_id;
    uint64_t section_length;

    uint64_t offsets_table_location;
    read_data_from_source(reinterpret_cast<char*>(&offsets_table_location), sizeof(offsets_table_location));

    // Also read the number of sections found in the region of volatile format
    uint64_t number_of_sections;
    read_data_from_source(reinterpret_cast<char*>(&number_of_sections), sizeof(number_of_sections));
    const size_t where_the_region_of_persistent_format_starts = get_cursor_position();

    // TODO bound checking
    move_cursor(offsets_table_location);
    read_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
    OPENVINO_ASSERT(section_id == PredefinedSectionID::OFFSETS_TABLE);

    read_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));
    m_parsed_sections[PredefinedSectionID::OFFSETS_TABLE] = OffsetsTableSection::read(this, section_length);
    m_offsets_table =
        std::dynamic_pointer_cast<OffsetsTableSection>(m_parsed_sections.at(PredefinedSectionID::OFFSETS_TABLE))
            ->get_table();

    // Step 2: Look for the CRE and evaluate it
    OPENVINO_ASSERT(m_offsets_table->count(PredefinedSectionID::CRE));
    move_cursor(m_offsets_table->at(PredefinedSectionID::CRE));

    read_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
    OPENVINO_ASSERT(section_id == PredefinedSectionID::CRE);

    read_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));
    m_parsed_sections[PredefinedSectionID::CRE] = CRESection::read(this, section_length);  // TODO also evaluate within
    std::dynamic_pointer_cast<CRESection>(m_parsed_sections.at(PredefinedSectionID::CRE))
        ->check_compatibility(plugin_capabilities_ids);

    // Step 3: Parse all known sections
    move_cursor(where_the_region_of_persistent_format_starts);

    while (number_of_sections--) {
        read_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
        read_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));

        if (!m_readers.count(section_id)) {
            // Unknown region, skip
            move_cursor(get_cursor_position() + section_length);
            continue;
        }

        m_parsed_sections[section_id] = m_readers.at(section_id)(this, section_length);
    }
}

}  // namespace intel_npu
