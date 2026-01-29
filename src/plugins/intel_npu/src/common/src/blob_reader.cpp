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
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor < m_npu_region_size);
    std::memcpy(destination, m_source.get().data<const char>() + m_cursor - size, size);
}

const void* BlobReader::interpret_data_from_source(const size_t size) {
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor < m_npu_region_size);
    return reinterpret_cast<const void*>(m_source.get().data<char>() + m_cursor - size);
}

ov::Tensor BlobReader::get_roi_tensor(const size_t size) {
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor < m_npu_region_size);
    return ov::Tensor(m_source, ov::Coordinate{m_cursor - size}, ov::Coordinate{m_cursor});
}

size_t BlobReader::get_cursor_relative_position() {
    return m_cursor;
}

void BlobReader::move_cursor_to_relative_position(const size_t offset) {
    OPENVINO_ASSERT(offset < m_npu_region_size);
    m_cursor = offset;
}

void BlobReader::read(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    copy_data_from_source(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    copy_data_from_source(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    // Read the size of the NPU region
    copy_data_from_source(reinterpret_cast<char*>(&m_npu_region_size), sizeof(m_npu_region_size));

    // Step 1: Read the table of offsets
    SectionID section_id;
    uint64_t section_length;

    uint64_t offsets_table_location;
    copy_data_from_source(reinterpret_cast<char*>(&offsets_table_location), sizeof(offsets_table_location));
    const size_t where_the_region_of_persistent_format_starts = get_cursor_relative_position();

    move_cursor_to_relative_position(offsets_table_location);
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
    OPENVINO_ASSERT(m_offsets_table.count(PredefinedSectionID::CRE),
                    "The CRE was not found within the table of offsets");
    move_cursor_to_relative_position(m_offsets_table.at(PredefinedSectionID::CRE));

    copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
    OPENVINO_ASSERT(section_id == PredefinedSectionID::CRE,
                    "Unexpected section ID. Expected: ",
                    PredefinedSectionID::CRE,
                    ". Received: ",
                    section_id);

    copy_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));
    m_parsed_sections[PredefinedSectionID::CRE] = CRESection::read(this, section_length);
    std::dynamic_pointer_cast<CRESection>(m_parsed_sections.at(PredefinedSectionID::CRE))
        ->get_cre()
        .check_compatibility(plugin_capabilities);

    // Step 3: Parse all known sections
    move_cursor_to_relative_position(where_the_region_of_persistent_format_starts);

    while (get_cursor_relative_position() < m_npu_region_size) {
        copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
        copy_data_from_source(reinterpret_cast<char*>(&section_length), sizeof(section_length));

        const size_t next_section_location = get_cursor_relative_position() + section_length;
        if (!m_readers.count(section_id)) {
            // Unknown region, skip
            move_cursor_to_relative_position(next_section_location);
            continue;
        }

        m_parsed_sections[section_id] = m_readers.at(section_id)(this, section_length);
        move_cursor_to_relative_position(next_section_location);  // jic the reader moved the cursor somewhere else
    }
}

std::optional<uint64_t> BlobReader::get_section_offset(const SectionID section_id) const {
    auto search_iterator = m_offsets_table.find(section_id);
    if (search_iterator == m_offsets_table.end()) {
        return std::nullopt;
    }

    return search_iterator->second;
}

size_t BlobReader::get_npu_region_size(std::istream& stream) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    stream.read(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    stream.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    uint64_t npu_region_size;
    stream.read(reinterpret_cast<char*>(&npu_region_size), sizeof(m_npu_region_size));

    return npu_region_size;
}

size_t BlobReader::get_npu_region_size(const ov::Tensor& tensor) {
    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    std::memcpy(const_cast<char*>(magic_bytes.c_str()), tensor.data<const char>(), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    std::memcpy(reinterpret_cast<char*>(&format_version),
                tensor.data<const char>() + MAGIC_BYTES.size(),
                sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    uint64_t npu_region_size;
    std::memcpy(reinterpret_cast<char*>(&npu_region_size),
                tensor.data<const char>() + MAGIC_BYTES.size() + sizeof(FORMAT_VERSION),
                sizeof(m_npu_region_size));

    return npu_region_size;
}

}  // namespace intel_npu
