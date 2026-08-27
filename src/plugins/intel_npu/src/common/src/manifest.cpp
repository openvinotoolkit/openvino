// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/manifest.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

Manifest::Manifest(const ov::log::Level log_level) : m_logger("Manifest", log_level) {}

void Manifest::add_entry(const SectionID id, const uint64_t offset, const uint64_t length) {
    OPENVINO_ASSERT(!m_table.count(id), "The section ID already exists within the manifest. ID: ", id);
    OPENVINO_ASSERT(!m_reversed_table.count(offset),
                    "The offset is already in-use within the manifest. Offset: ",
                    offset,
                    ". ID: ",
                    id);

    m_logger.debug("New entry added: section ID %s offset %lu, length %lu", id.to_string(), offset, length);

    m_table[id] = std::make_pair<>(offset, length);
    m_reversed_table[offset] = id;
}

size_t Manifest::get_entry_size() {
    // Type ID, instance ID, offset, length
    return sizeof(SectionType) + sizeof(SectionTypeInstance) + 2 * sizeof(uint64_t);
}

std::optional<uint64_t> Manifest::lookup_offset(const SectionID id) const {
    const auto search_result = m_table.find(id);
    if (search_result != m_table.end()) {
        return search_result->second.first;
    }
    return std::nullopt;
}

std::optional<uint64_t> Manifest::lookup_length(const SectionID id) const {
    const auto search_result = m_table.find(id);
    if (search_result != m_table.end()) {
        return search_result->second.second;
    }
    return std::nullopt;
}

std::optional<SectionID> Manifest::lookup_section_id(const uint64_t offset) const {
    const auto search_result = m_reversed_table.find(offset);
    if (search_result != m_reversed_table.end()) {
        return search_result->second;
    }
    return std::nullopt;
}

size_t Manifest::get_number_of_entries() const {
    return m_table.size();
}

std::unordered_set<SectionID> Manifest::get_all_registered_section_ids() const {
    std::unordered_set<SectionID> ids;

    for (const auto& [key, value] : m_table) {
        ids.insert(key);
    }
    return ids;
}

bool Manifest::empty() const {
    return m_table.empty();
}

ManifestSection::ManifestSection(const Manifest& manifest, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::MANIFEST),
      m_manifest(manifest),
      m_logger("ManifestSection", log_level) {}

void ManifestSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ManifestSection::write");

    m_logger.debug("Writting %lu entries", m_manifest.get_number_of_entries());

    for (const auto& [key, value] : m_manifest.m_table) {
        // Section type ID, Section instanfce type ID, offset, length
        writer.write_from(&key.type, sizeof(key.type));
        writer.write_from(&key.type_instance, sizeof(key.type_instance));
        writer.write_from(&value.first, sizeof(value.first));
        writer.write_from(&value.second, sizeof(value.second));

        m_logger.trace("Entry written: section ID %s, offset %lu, length %lu",
                       key.to_string(),
                       value.first,
                       value.second);
    }
}

Manifest ManifestSection::get_table() const {
    return m_manifest;
}

std::shared_ptr<ISection> ManifestSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ManifestSection::read");
    Logger logger("ManifestSection", blob_reader.get_log_level());

    const size_t section_length = blob_reader.get_section_length();
    const size_t entry_size = Manifest::get_entry_size();
    OPENVINO_ASSERT(
        section_length % entry_size == 0,
        "Received a manifest section length that is not divisible by the table entry size. Section length: ",
        section_length,
        ". Table entry size: ",
        entry_size);

    size_t number_of_sections_in_table = section_length / entry_size;
    Manifest manifest(blob_reader.get_log_level());
    SectionType type;
    SectionTypeInstance type_instance;
    uint64_t offset;
    uint64_t length;

    logger.debug("Reading %lu entries", number_of_sections_in_table);

    while (number_of_sections_in_table--) {
        blob_reader.read_into_buffer(&type, sizeof(type));
        blob_reader.read_into_buffer(&type_instance, sizeof(type_instance));
        blob_reader.read_into_buffer(&offset, sizeof(offset));
        blob_reader.read_into_buffer(&length, sizeof(length));

        const SectionID section_id(type, type_instance);
        manifest.add_entry(section_id, offset, length);

        logger.trace("Read entry: section ID %s, offset %lu, length %lu", section_id.to_string(), offset, length);
    }

    return std::make_shared<ManifestSection>(std::move(manifest), logger.level());
}

}  // namespace intel_npu
