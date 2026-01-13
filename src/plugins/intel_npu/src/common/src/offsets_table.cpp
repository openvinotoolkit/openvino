// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/offsets_table.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace intel_npu {

OffsetsTableSection::OffsetsTableSection(const std::shared_ptr<std::unordered_map<SectionID, uint64_t>>& offsets_table)
    : ISection(PredefinedSectionID::OFFSETS_TABLE),
      m_offsets_table(offsets_table) {}

void OffsetsTableSection::write(std::ostream& stream, BlobWriter* writer) {
    for (const auto& [key, value] : *m_offsets_table) {
        stream.write(reinterpret_cast<const char*>(&key), sizeof(key));
        stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

std::optional<uint64_t> OffsetsTableSection::get_length() const {
    return m_offsets_table->size() * (sizeof(SectionID) + sizeof(uint64_t));
}

std::shared_ptr<std::unordered_map<SectionID, uint64_t>> OffsetsTableSection::get_table() const {
    return m_offsets_table;
}

std::shared_ptr<ISection> OffsetsTableSection::read(BlobReader* blob_reader, const size_t section_length) {
    auto offsets_table = std::make_shared<std::unordered_map<SectionID, uint64_t>>();
    size_t number_of_sections_in_table = section_length / (sizeof(SectionID) + sizeof(uint64_t));
    SectionID section_id;
    uint64_t offset;

    while (number_of_sections_in_table--) {
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&section_id), sizeof(section_id));
        blob_reader->copy_data_from_source(reinterpret_cast<char*>(&offset), sizeof(offset));
        (*offsets_table)[section_id] = offset;
    }

    return std::make_shared<OffsetsTableSection>(offsets_table);
}

}  // namespace intel_npu
