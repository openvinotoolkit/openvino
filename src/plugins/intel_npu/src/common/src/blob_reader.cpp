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

BlobReader::BlobReader(const ov::Tensor& source) : m_source(source), m_cursor(0) {
    // Register the core sections
    register_reader(PredefinedSectionID::CRE, CRESection::read);
    register_reader(PredefinedSectionID::OFFSETS_TABLE, OffsetsTableSection::read);
}

void BlobReader::register_reader(const SectionID section_id,
                                 std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader) {
    m_readers[section_id] = reader;
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID section_id) {
    auto search_result = m_parsed_sections.find(section_id);
    if (search_result != m_parsed_sections.end()) {
        return search_result->second;
    }
    return nullptr;
}

void BlobReader::copy_data_from_source(char* destination, const size_t size) {
    std::memcpy(destination, m_source.get().data<const char>() + m_cursor, size);
    m_cursor += size;
}

const void* BlobReader::interpret_data_from_source(const size_t size) {
    m_cursor += size;
    return reinterpret_cast<const void*>(m_source.get().data<char>() + m_cursor - size);
}

ov::Tensor BlobReader::get_roi_tensor(const size_t size) {
    m_cursor += size;
    return ov::Tensor(m_source, ov::Coordinate{m_cursor - size}, ov::Coordinate{m_cursor});
}

size_t BlobReader::get_cursor_position() {
    return m_cursor;
}

void BlobReader::move_cursor(const size_t offset) {
    m_cursor = offset;
}

void BlobReader::read(const std::unordered_set<CRE::Token>& plugin_capabilities_ids) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    copy_data_from_source(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    copy_data_from_source(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    // Read the size of the NPU region
    uint64_t npu_region_size;
    copy_data_from_source(reinterpret_cast<char*>(&npu_region_size), sizeof(npu_region_size));

    // Step 1: Read the table of offsets
    SectionID section_id;
    uint64_t section_length;

    uint64_t offsets_table_location;
    copy_data_from_source(reinterpret_cast<char*>(&offsets_table_location), sizeof(offsets_table_location));
    const size_t where_the_region_of_persistent_format_starts = get_cursor_position();

    // TODO bound checking
    move_cursor(offsets_table_location);
    copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
    OPENVINO_ASSERT(section_id == PredefinedSectionID::OFFSETS_TABLE,
                    "Unexpected section ID. Expected: ",
                    PredefinedSectionID::OFFSETS_TABLE,
                    ". Received: ",
                    section_id);

    copy_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));
    m_parsed_sections[PredefinedSectionID::OFFSETS_TABLE] = OffsetsTableSection::read(this, section_length);
    m_offsets_table =
        std::dynamic_pointer_cast<OffsetsTableSection>(m_parsed_sections.at(PredefinedSectionID::OFFSETS_TABLE))
            ->get_table();

    // Step 2: Look for the CRE and evaluate it
    OPENVINO_ASSERT(m_offsets_table->count(PredefinedSectionID::CRE),
                    "The CRE was not found within the table of offsets");
    move_cursor(m_offsets_table->at(PredefinedSectionID::CRE));

    copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
    OPENVINO_ASSERT(section_id == PredefinedSectionID::CRE,
                    "Unexpected section ID. Expected: ",
                    PredefinedSectionID::CRE,
                    ". Received: ",
                    section_id);

    copy_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));
    m_parsed_sections[PredefinedSectionID::CRE] = CRESection::read(this, section_length);
    std::dynamic_pointer_cast<CRESection>(m_parsed_sections.at(PredefinedSectionID::CRE))
        ->check_compatibility(plugin_capabilities_ids);

    // Step 3: Parse all known sections
    move_cursor(where_the_region_of_persistent_format_starts);

    while (get_cursor_position() < npu_region_size) {
        copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
        copy_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));

        if (!m_readers.count(section_id)) {
            // Unknown region, skip
            move_cursor(get_cursor_position() + section_length);
            continue;
        }

        m_parsed_sections[section_id] = m_readers.at(section_id)(this, section_length);
    }
}

size_t BlobReader::get_npu_region_size(std::istream& stream) {
    uint64_t npu_region_size;
    auto position_before = stream.tellg();

    // Magic bytes -> format version -> NPU region size
    stream.seekg(MAGIC_BYTES.size() + sizeof(FORMAT_VERSION), std::ios_base::cur);
    stream.read(reinterpret_cast<char*>(&npu_region_size), sizeof(npu_region_size));
    stream.seekg(position_before);

    return npu_region_size;
}

size_t BlobReader::get_npu_region_size(const ov::Tensor& tensor) {
    uint64_t npu_region_size;

    // Magic bytes -> format version -> NPU region size
    std::memcpy(reinterpret_cast<char*>(&npu_region_size),
                tensor.data<const char>() + MAGIC_BYTES.size() + sizeof(FORMAT_VERSION),
                sizeof(npu_region_size));

    return npu_region_size;
}

}  // namespace intel_npu
