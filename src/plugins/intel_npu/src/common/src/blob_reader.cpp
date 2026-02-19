// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

constexpr intel_npu::SectionTypeInstance FIRST_INSTANCE_ID = 0;

}  // namespace

namespace intel_npu {

BlobReader::BlobReader(const ov::Tensor& source) : m_source(source), m_cursor(0) {
    // Register the core sections
    register_reader(PredefinedSectionType::CRE, CRESection::read);
    register_reader(PredefinedSectionType::OFFSETS_TABLE, OffsetsTableSection::read);
}

void BlobReader::register_reader(const SectionType type,
                                 std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader) {
    m_readers[type] = reader;
}

std::shared_ptr<ISection> BlobReader::retrieve_section(const SectionID& id) {
    auto type_search_result = m_parsed_sections.find(id.type);
    if (type_search_result != m_parsed_sections.end()) {
        auto instance_search_result = type_search_result->second.find(id.type_instance);
        if (instance_search_result != type_search_result->second.end()) {
            return instance_search_result->second;
        }
    }
    return nullptr;
}

std::shared_ptr<ISection> BlobReader::retrieve_first_section(const SectionType section_type) {
    return retrieve_section(SectionID(section_type, FIRST_INSTANCE_ID));
}

std::optional<std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>
BlobReader::retrieve_sections_same_type(const SectionType type) {
    auto type_search_result = m_parsed_sections.find(type);
    if (type_search_result != m_parsed_sections.end()) {
        return type_search_result->second;
    }
    return std::nullopt;
}

void BlobReader::copy_data_from_source(char* destination, const size_t size) {
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_npu_region_size);
    std::memcpy(destination, m_source.get().data<const char>() + m_cursor - size, size);
}

const void* BlobReader::interpret_data_from_source(const size_t size) {
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_npu_region_size);
    return reinterpret_cast<const void*>(m_source.get().data<char>() + m_cursor - size);
}

ov::Tensor BlobReader::get_roi_tensor(const size_t size) {
    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_npu_region_size);
    return ov::Tensor(m_source, ov::Coordinate{m_cursor - size}, ov::Coordinate{m_cursor});
}

size_t BlobReader::get_cursor_relative_position() {
    return m_cursor;
}

void BlobReader::move_cursor_to_relative_position(const size_t offset) {
    OPENVINO_ASSERT(offset <= m_npu_region_size);
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
    uint64_t offsets_table_location;
    uint64_t offsets_table_size;
    copy_data_from_source(reinterpret_cast<char*>(&offsets_table_location), sizeof(offsets_table_location));
    copy_data_from_source(reinterpret_cast<char*>(&offsets_table_size), sizeof(offsets_table_size));

    const size_t where_the_region_of_persistent_format_starts = get_cursor_relative_position();

    move_cursor_to_relative_position(offsets_table_location);

    m_parsed_sections[PredefinedSectionType::OFFSETS_TABLE][FIRST_INSTANCE_ID] =
        OffsetsTableSection::read(this, offsets_table_size);
    m_parsed_sections[PredefinedSectionType::OFFSETS_TABLE][FIRST_INSTANCE_ID]->set_section_type_instance(
        FIRST_INSTANCE_ID);
    m_offsets_table = std::dynamic_pointer_cast<OffsetsTableSection>(
                          m_parsed_sections.at(PredefinedSectionType::OFFSETS_TABLE).at(FIRST_INSTANCE_ID))
                          ->get_table();

    // Step 2: Look for the CRE and evaluate it
    std::optional<uint64_t> offset = m_offsets_table.lookup_offset(CRE_SECTION_ID);
    std::optional<uint64_t> section_length = m_offsets_table.lookup_length(CRE_SECTION_ID);
    OPENVINO_ASSERT(offset.has_value(), "The CRE was not found within the table of offsets");
    move_cursor_to_relative_position(offset.value());

    m_parsed_sections[PredefinedSectionType::CRE][FIRST_INSTANCE_ID] = CRESection::read(this, section_length.value());
    m_parsed_sections[PredefinedSectionType::CRE][FIRST_INSTANCE_ID]->set_section_type_instance(FIRST_INSTANCE_ID);
    std::dynamic_pointer_cast<CRESection>(m_parsed_sections.at(PredefinedSectionType::CRE).at(FIRST_INSTANCE_ID))
        ->get_cre()
        .check_compatibility(plugin_capabilities);

    // Step 3: Parse all known sections
    move_cursor_to_relative_position(where_the_region_of_persistent_format_starts);

    size_t relative_offset;
    while (relative_offset = get_cursor_relative_position(), relative_offset < m_npu_region_size) {
        if (relative_offset == offsets_table_location) {
            move_cursor_to_relative_position(relative_offset + offsets_table_size);
            continue;
        }

        const std::optional<SectionID> section_id = m_offsets_table.lookup_section_id(relative_offset);
        OPENVINO_ASSERT(section_id.has_value(),
                        "Did not find any section corresponding to the relative offset ",
                        relative_offset);
        section_length = m_offsets_table.lookup_length(section_id.value());

        const size_t next_section_location = relative_offset + section_length.value();

        // Read the section if we have a reader for it. Otherwise, skip it.
        if (m_readers.count(section_id.value().type)) {
            m_parsed_sections[section_id.value().type][section_id.value().type_instance] =
                m_readers.at(section_id.value().type)(this, section_length.value());
            m_parsed_sections[section_id.value().type][section_id.value().type_instance]->set_section_type_instance(
                section_id.value().type_instance);
        }

        move_cursor_to_relative_position(next_section_location);
    }
}

size_t BlobReader::get_npu_region_size(std::istream& stream) {
    const auto cursor_before_reading = stream.tellg();

    std::string magic_bytes(MAGIC_BYTES.size(), 0);
    stream.read(const_cast<char*>(magic_bytes.c_str()), MAGIC_BYTES.size());
    OPENVINO_ASSERT(magic_bytes == MAGIC_BYTES);

    uint32_t format_version;
    stream.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    OPENVINO_ASSERT(format_version == FORMAT_VERSION);

    uint64_t npu_region_size;
    stream.read(reinterpret_cast<char*>(&npu_region_size), sizeof(m_npu_region_size));
    stream.seekg(cursor_before_reading);

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
