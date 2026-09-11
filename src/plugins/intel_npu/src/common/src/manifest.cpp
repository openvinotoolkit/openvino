// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/manifest.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

Manifest::Manifest(const ov::log::Level log_level) : m_logger("Manifest", log_level) {}

void Manifest::add_entry(const SectionID id, const SectionType type, const uint64_t offset, const uint64_t length) {
    const std::string debug_name = section_type_and_id_to_string(type, id);
    OPENVINO_ASSERT(!m_id_to_attributes.count(id), "The section ", debug_name, " already exists within the manifest.");
    OPENVINO_ASSERT(!m_offset_to_id.count(offset),
                    "The offset is already in-use within the manifest. Offset: ",
                    offset,
                    ". Section: ",
                    debug_name);

    m_logger.debug("New entry added: section %s, offset %zu, length %zu", debug_name.data(), offset, length);

    m_id_to_attributes.emplace(id, std::make_tuple(type, offset, length));
    m_offset_to_id.emplace(offset, id);

    if (m_type_to_ids.count(type)) {
        m_type_to_ids.at(type).push_back(id);
        return;
    }
    m_type_to_ids.emplace(type, std::vector<SectionID>{id});
}

size_t Manifest::get_entry_size() {
    // ID, type, offset, length
    return 2 * sizeof(uint16_t) + 2 * sizeof(uint64_t);
}

// TODO minor refactor?
std::optional<SectionType> Manifest::lookup_type(const SectionID id) const {
    const auto search_result = m_id_to_attributes.find(id);
    return search_result != m_id_to_attributes.end() ? std::make_optional<>(std::get<0>(search_result->second))
                                                     : std::nullopt;
}

std::optional<uint64_t> Manifest::lookup_offset(const SectionID id) const {
    const auto search_result = m_id_to_attributes.find(id);
    return search_result != m_id_to_attributes.end() ? std::make_optional<>(std::get<1>(search_result->second))
                                                     : std::nullopt;
}

std::optional<uint64_t> Manifest::lookup_length(const SectionID id) const {
    const auto search_result = m_id_to_attributes.find(id);
    return search_result != m_id_to_attributes.end() ? std::make_optional<>(std::get<2>(search_result->second))
                                                     : std::nullopt;
}

std::optional<SectionID> Manifest::lookup_section_id(const uint64_t offset) const {
    const auto search_result = m_offset_to_id.find(offset);
    if (search_result != m_offset_to_id.end()) {
        return search_result->second;
    }
    return std::nullopt;
}

std::vector<SectionID> Manifest::lookup_section_ids(const SectionType type) const {
    if (!m_type_to_ids.count(type)) {
        return std::vector<SectionID>();
    }
    return m_type_to_ids.at(type);
}

size_t Manifest::get_number_of_entries() const {
    return m_id_to_attributes.size();
}

std::unordered_set<SectionID> Manifest::get_all_registered_section_ids() const {
    std::unordered_set<SectionID> ids;

    for (const auto& [key, value] : m_id_to_attributes) {
        ids.insert(key);
    }
    return ids;
}

bool Manifest::empty() const {
    return m_id_to_attributes.empty();
}

ManifestSection::ManifestSection(const Manifest& manifest, const ov::log::Level log_level)
    : ISection(SectionTypeCode::MANIFEST),
      m_manifest(manifest),
      m_logger("ManifestSection", log_level) {}

void ManifestSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "ManifestSection::write");

    m_logger.debug("Writting %lu entries", m_manifest.get_number_of_entries());

    for (const auto& [id, values] : m_manifest.m_id_to_attributes) {
        const auto [type, offset, length] = values;
        const uint16_t id_value = id.get_id();
        const uint16_t type_code = static_cast<uint16_t>(type.get_code());

        // ID, type, offset, length
        writer.write_from(&id_value, sizeof(id_value));
        writer.write_from(&type_code, sizeof(type_code));
        writer.write_from(&offset, sizeof(offset));
        writer.write_from(&length, sizeof(length));

        m_logger.trace("Entry written: section %s, offset %lu, length %lu",
                       section_type_and_id_to_string(type, id),
                       offset,
                       length);
    }
}

Manifest ManifestSection::get_manifest() const {
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
    uint16_t id;
    SectionTypeCode type;
    uint64_t offset;
    uint64_t length;

    logger.debug("Reading %lu entries", number_of_sections_in_table);

    while (number_of_sections_in_table--) {
        blob_reader.read_into_buffer(&id, sizeof(id));
        blob_reader.read_into_buffer(&type, sizeof(type));
        blob_reader.read_into_buffer(&offset, sizeof(offset));
        blob_reader.read_into_buffer(&length, sizeof(length));

        manifest.add_entry(id, type, offset, length);

        logger.trace("Read the entry: section %s, offset %lu, length %lu",
                     section_type_and_id_to_string(type, id).data(),
                     offset,
                     length);
    }

    return std::make_shared<ManifestSection>(std::move(manifest), logger.level());
}

}  // namespace intel_npu
