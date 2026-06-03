// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/offsets_table.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

OffsetsTable::OffsetsTable(const ov::log::Level log_level) : m_logger("OffsetsTable", log_level) {}

void OffsetsTable::add_entry(const SectionID id, const uint64_t offset, const uint64_t length) {
    // TODO maybe add some message when failing
    // "Section ID already existing in the table: printf(id)"
    OPENVINO_ASSERT(!m_table.count(id), "The section ID already exists within the table of offsets. ID: ", id);
    OPENVINO_ASSERT(!m_reversed_table.count(offset),
                    "The offset is already in-use within the table of offsets. Offset: ",
                    offset,
                    ". ID: ",
                    id);

    m_logger.debug("New entry added: section ID (%lu, %lu), offset %lu, length %lu",
                   id.type,
                   id.type_instance,
                   offset,
                   length);

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

size_t OffsetsTable::get_number_of_entries() const {
    return m_table.size();
}

bool OffsetsTable::empty() const {
    return m_table.empty();
}

OffsetsTableSection::OffsetsTableSection(const OffsetsTable& offsets_table, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::OFFSETS_TABLE),
      m_offsets_table(offsets_table),
      m_logger("OffsetsTableSection", log_level) {}

void OffsetsTableSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "OffsetsTableSection::write");

    m_logger.debug("Writting %lu entries", m_offsets_table.get_number_of_entries());

    for (const auto& [key, value] : m_offsets_table.m_table) {
        // Section type ID, Section instanfce type ID, offset, length
        writer.write(&key.type, sizeof(key.type));
        writer.write(&key.type_instance, sizeof(key.type_instance));
        writer.write(&value.first, sizeof(value.first));
        writer.write(&value.second, sizeof(value.second));

        m_logger.trace("Entry written: section ID (%lu, %lu), offset %lu, length %lu",
                       key.type,
                       key.type_instance,
                       value.first,
                       value.second);
    }
}

OffsetsTable OffsetsTableSection::get_table() const {
    return m_offsets_table;
}

std::shared_ptr<ISection> OffsetsTableSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "OffsetsTableSection::read");
    Logger logger("OffsetsTableSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    const size_t entry_size = OffsetsTable::get_entry_size();
    OPENVINO_ASSERT(
        section_length % entry_size == 0,
        "Received an offsets table section length that is not divisible by the table entry size. Section length: ",
        section_length,
        ". Table entry size: ",
        entry_size);

    size_t number_of_sections_in_table = section_length / entry_size;
    OffsetsTable offsets_table(blob_reader.get_log_level());
    SectionType type;
    SectionTypeInstance type_instance;
    uint64_t offset;
    uint64_t length;

    logger.debug("Reading %lu entries", number_of_sections_in_table);

    while (number_of_sections_in_table--) {
        blob_reader.copy_data_from_source(reinterpret_cast<char*>(&type), sizeof(type));
        blob_reader.copy_data_from_source(reinterpret_cast<char*>(&type_instance), sizeof(type_instance));
        blob_reader.copy_data_from_source(reinterpret_cast<char*>(&offset), sizeof(offset));
        blob_reader.copy_data_from_source(reinterpret_cast<char*>(&length), sizeof(length));
        offsets_table.add_entry(SectionID(type, type_instance), offset, length);

        logger.trace("Read entry: section ID (%lu, %lu), offset %lu, length %lu", type, type_instance, offset, length);
    }

    return std::make_shared<OffsetsTableSection>(std::move(offsets_table), logger.level());
}

}  // namespace intel_npu
