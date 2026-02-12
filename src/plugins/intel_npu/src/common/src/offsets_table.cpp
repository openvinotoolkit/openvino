// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/offsets_table.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace intel_npu {

void OffsetsTable::add_entry(const SectionID id, const uint64_t offset, const uint64_t length) {
    OPENVINO_ASSERT(!m_table.count(id));
    OPENVINO_ASSERT(!m_reversed_table.count(offset));

    m_table[id] = std::make_pair<>(offset, length);
    m_reversed_table[offset] = id;
}

size_t OffsetsTable::get_entry_size() {
    // Type ID, instance ID, offset, length
    return sizeof(SectionType) + sizeof(SectionTypeInstance) + 2 * sizeof(uint64_t);
}

std::optional<uint64_t> OffsetsTable::lookup_offset(const SectionID id) const {
    const auto search_result = m_table.find(id);
    if (search_result != m_table.end()) {
        return search_result->second.first;
    }
    return std::nullopt;
}

std::optional<uint64_t> OffsetsTable::lookup_length(const SectionID id) const {
    const auto search_result = m_table.find(id);
    if (search_result != m_table.end()) {
        return search_result->second.second;
    }
    return std::nullopt;
}

std::optional<SectionID> OffsetsTable::lookup_section_id(const uint64_t offset) const {
    const auto search_result = m_reversed_table.find(offset);
    if (search_result != m_reversed_table.end()) {
        return search_result->second;
    }
    return std::nullopt;
}

OffsetsTableSection::OffsetsTableSection(const OffsetsTable& offsets_table)
    : ISection(PredefinedSectionType::OFFSETS_TABLE),
      m_offsets_table(offsets_table) {}

void OffsetsTableSection::write(std::ostream& stream, BlobWriter* writer) {
    for (const auto& [key, value] : m_offsets_table.m_table) {
        // Section type ID, Section instanfce type ID, offset, length
        stream.write(reinterpret_cast<const char*>(&key.type), sizeof(key.type));
        stream.write(reinterpret_cast<const char*>(&key.type_instance), sizeof(key.type_instance));
        stream.write(reinterpret_cast<const char*>(&value.first), sizeof(value.first));
        stream.write(reinterpret_cast<const char*>(&value.second), sizeof(value.second));
    }
}

OffsetsTable OffsetsTableSection::get_table() const {
    return m_offsets_table;
}

std::shared_ptr<ISection> OffsetsTableSection::read(BlobReader* blob_reader, const size_t section_length) {
    OffsetsTable offsets_table;
    size_t number_of_sections_in_table = section_length / offsets_table.get_entry_size();
    SectionType type;
    SectionTypeInstance type_instance;
    uint64_t offset;
    uint64_t length;

    while (number_of_sections_in_table--) {
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&type), sizeof(type));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&type_instance), sizeof(type_instance));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&offset), sizeof(offset));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&length), sizeof(length));
        offsets_table.add_entry(SectionID(type, type_instance), offset, length);
    }

    return std::make_shared<OffsetsTableSection>(std::move(offsets_table));
}

}  // namespace intel_npu
