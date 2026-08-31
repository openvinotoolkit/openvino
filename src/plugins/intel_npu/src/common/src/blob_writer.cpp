// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_writer.hpp"

#include <iterator>

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/itt.hpp"
#include "intel_npu/common/runtime_requirements_section.hpp"

namespace {

constexpr std::string_view MAGIC_BYTES = "OVNPU";
constexpr uint32_t FORMAT_VERSION = 0x30000;  // 3.0;

constexpr size_t FIRST_INSTANCE_ID = 0;

}  // namespace

namespace intel_npu {

BlobWriterInterface::BlobWriterInterface(std::ostream& stream,
                                         const std::streampos stream_npu_region_start,
                                         const ov::log::Level log_level)
    : m_stream(stream),
      m_stream_npu_region_start(stream_npu_region_start),
      m_stream_current_section_start(stream.tellp()),
      m_logger("BlobWriterInterface", log_level) {
    m_logger.debug("Created a new BlobWriterInterface. Section start: %lu", m_stream_current_section_start);
}

void BlobWriterInterface::write_from(const void* source, const size_t size) {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"write\" operation");
    m_logger.trace("Writing %lu bytes", size);
    m_stream.get().write(reinterpret_cast<const char*>(source), size);
}

void BlobWriterInterface::add_padding(const size_t size) {
    m_logger.trace("Adding %lu bytes of padding", size);
    if (size > 0) {
        std::fill_n(std::ostream_iterator<char>(m_stream.get()), size, 0);
    }
}

std::streamoff BlobWriterInterface::get_offset_relative_to_current_section() const {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"tell\" operation");
    return m_stream.get().tellp() - m_stream_current_section_start;
}

void BlobWriterInterface::move_cursor_relative_to_current_section(const size_t offset) {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"seek\" operation");
    m_stream.get().seekp(m_stream_current_section_start + static_cast<std::streamoff>(offset));
    // This check will fail if the destination goes beyond the end of the stream
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream after \"seek\" operation");
}

std::streamoff BlobWriterInterface::get_offset_relative_to_npu_region() const {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"tell\" operation");
    return m_stream.get().tellp() - m_stream_npu_region_start;
}

void BlobWriterInterface::move_cursor_relative_to_npu_region(const size_t offset) {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"seek\" operation");
    OPENVINO_ASSERT(m_stream_current_section_start >= m_stream_npu_region_start,
                    "Invalid section start. The beginning of a section should be placed within the stream region "
                    "dedicated to the NPU plugin.");
    OPENVINO_ASSERT(offset >= static_cast<size_t>(m_stream_current_section_start - m_stream_npu_region_start),
                    "A section writer has attempted a jump outside the boundaries of its own payload");
    m_stream.get().seekp(m_stream_npu_region_start + static_cast<std::streamoff>(offset));
    // This check will fail if the destination goes beyond the end of the stream
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"seek\" operation");
}

void BlobWriterInterface::seek_to_the_end() {
    OPENVINO_ASSERT(m_stream.get().good(), "Invalid stream before \"seek to the end\" operation");
    m_stream.get().seekp(0, std::ios_base::end);
}

BlobWriter::BlobWriter(const ov::log::Level log_level)
    : m_next_section_id(FIRST_INSTANCE_ID),
      m_logger("BlobWriter", log_level) {
    m_logger.debug("BlobWriter built from scratch");
}

BlobWriter::BlobWriter(const std::shared_ptr<BlobReader>& blob_reader, const ov::log::Level log_level)
    : m_next_section_id(FIRST_INSTANCE_ID),
      m_logger("BlobWriter", log_level) {
    m_logger.debug("Building the BlobWriter using the contents of a BlobReader");

    for (const SectionID& section_id : blob_reader->m_parsed_sections_order) {
        // The CRE & manifest sections are added by the write() method after writing all registered sections
        // (jic the registered sections will alter the CRE/table). Therefore, these sections should be omitted here.
        OPENVINO_ASSERT(
            section_id.type != PredefinedSectionType::MANIFEST && section_id.type != PredefinedSectionType::CRE,
            "By convention, the manifest and CRE sections should not be found within the parsed sections order "
            "attribute");
        register_section_from_blob_reader(blob_reader->retrieve_section(section_id));
        m_logger.debug("Registered section ID %s", section_id.to_string());
    }
}

std::streamoff BlobWriter::get_offset_relative_to_npu_region(std::ostream& stream,
                                                             const std::streampos stream_npu_region_start) const {
    OPENVINO_ASSERT(stream.good(), "Invalid stream before \"tell\" operation");
    return stream.tellp() - stream_npu_region_start;
}

SectionID BlobWriter::register_section(const std::shared_ptr<ISection>& section) {
    const SectionType section_type = section->get_section_type();

    // TODO overflow checks
    const SectionID section_id = m_next_section_id++;
    section->set_section_type_instance(section_id);
    m_write_queue.push(section);

    if (m_registered_sections.count(section_type)) {
        OPENVINO_ASSERT(!m_registered_sections.at(section_type).count(section_id),
                        "The same section ID has been attributed to two distinct sections");
    }
    m_registered_sections[section_type][section_id] = section;

    m_logger.debug("Registered section %s", section->get_section_id()->to_string());

    return section_id;
}

void BlobWriter::register_section_from_blob_reader(const std::shared_ptr<ISection>& section) {
    const SectionType section_type = section->get_section_type();

    // Update the next instance ID to be used.
    OPENVINO_ASSERT(section->get_section_id().has_value(),
                    "Found a section parsed by a BlobReader object without an ID");
    const SectionID candidate = section->get_section_id().value() + 1;
    m_next_section_id = candidate > m_next_section_id ? candidate : m_next_section_id;

    m_write_queue.push(section);

    if (m_registered_sections.count(section_type)) {
        OPENVINO_ASSERT(!m_registered_sections.at(section_type).count(section->get_section_type_instance().value()),
                        "The same section ID has been attributed to two distinct sections");
    }
    m_registered_sections[section_type][section->get_section_type_instance().value()] = section;
}

size_t BlobWriter::count_registered_sections_of_type(const SectionType type) const {
    return m_registered_sections.count(type) ? m_registered_sections.at(type).size() : 0;
}

CRE BlobWriter::build_cre() const {
    m_logger.debug("Filling the CRE");

    CRE cre(m_logger.level());
    m_logger.debug("Added the CRE_EVALUATION token to the CRE");

    // Go through all sections to find out the tokens that are needed
    for (const auto& [section_type, instance_map] : m_registered_sections) {
        for (const auto& [section_instance, section] : instance_map) {
            cre.append_to_expression(section->get_compatibility_requirements_subexpression(m_registered_sections));
        }
    }

    return cre;
}

void BlobWriter::write_section(std::ostream& stream,
                               const std::shared_ptr<ISection>& section,
                               const std::streampos stream_npu_region_start,
                               Manifest& manifest) const {
    const std::optional<SectionID> section_id = section->get_section_id();
    OPENVINO_ASSERT(section_id.has_value(), "Missing section ID while writing the section");
    m_logger.debug("Writting the section identified as %s", section_id->to_string());

    stream.seekp(0, std::ios_base::end);
    const uint64_t offset = get_offset_relative_to_npu_region(stream, stream_npu_region_start);
    BlobWriterInterface blob_writer_interface(stream, stream_npu_region_start, m_logger.level());

    section->write(blob_writer_interface);

    stream.seekp(0, std::ios_base::end);
    const uint64_t length = static_cast<uint64_t>(blob_writer_interface.get_offset_relative_to_npu_region() - offset);

    // All sections registered within the BlobWriter are automatically added to the manifest
    // The instance ID should have been added by the writer. Therefore, the section ID should exist.
    manifest.add_entry(section_id.value(), offset, length);
}

void BlobWriter::write_to(std::ostream& stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BlobWriter::write");
    m_logger.debug("Starting to write to a stream");

    // Operate on this copy instead of the attribute. This is necessary to ensure write idempotency by keeping the
    // attributes unchanged.
    std::queue<std::shared_ptr<ISection>> write_queue = m_write_queue;
    const std::streampos stream_npu_region_start = stream.tellp();

    // The manifest corresponds to a single blob written into a stream. Therefore, this object should exist
    // only within the scope of the writing session.
    Manifest manifest(m_logger.level());

    // The header
    stream.write(reinterpret_cast<const char*>(MAGIC_BYTES.data()), MAGIC_BYTES.size());
    stream.write(reinterpret_cast<const char*>(&FORMAT_VERSION), sizeof(FORMAT_VERSION));

    // Stop condition for the BlobReader: the size of the data written here
    const auto will_come_back_to_this_at_the_end = stream.tellp();
    uint64_t npu_region_size = 0;
    stream.write(reinterpret_cast<const char*>(&npu_region_size),
                 sizeof(npu_region_size));  // placeholder

    // Placeholder until the manifest is fully populated and written into the blob
    uint64_t manifest_location = 0;
    uint64_t manifest_size = 0;
    stream.write(reinterpret_cast<const char*>(&manifest_location), sizeof(manifest_location));
    stream.write(reinterpret_cast<const char*>(&manifest_size), sizeof(manifest_size));

    // The region of dynamic format (list of key-length-payload sections, any order & no restrictions w.r.t.
    // the content of the payload)

    // Write the RuntimeRequirementsSection. This section doesn't have to be the first one, but we write it first to
    // emphasize the fact that section writers cannot append to the "global" CRE
    const auto cre_section = std::make_shared<RuntimeRequirementsSection>(build_cre(), m_logger.level());
    cre_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(stream, cre_section, stream_npu_region_start, manifest);

    while (!write_queue.empty()) {
        const std::shared_ptr<ISection>& section = write_queue.front();
        write_queue.pop();

        write_section(stream, section, stream_npu_region_start, manifest);
    }

    // Write the manifest
    manifest_location = get_offset_relative_to_npu_region(stream, stream_npu_region_start);

    const auto manifest_section = std::make_shared<ManifestSection>(manifest, m_logger.level());
    manifest_section->set_section_type_instance(FIRST_INSTANCE_ID);
    write_section(stream, manifest_section, stream_npu_region_start, manifest);

    npu_region_size = get_offset_relative_to_npu_region(stream, stream_npu_region_start);
    manifest_size = npu_region_size - manifest_location;

    // Go back to the beginning and write the size of the whole NPU region & the location of the manifest
    stream.seekp(will_come_back_to_this_at_the_end);
    stream.write(reinterpret_cast<const char*>(&npu_region_size), sizeof(npu_region_size));
    stream.write(reinterpret_cast<const char*>(&manifest_location), sizeof(manifest_location));
    stream.write(reinterpret_cast<const char*>(&manifest_size), sizeof(manifest_size));

    m_logger.trace("NPU region size %lu", npu_region_size);
    m_logger.trace("Manifest location %lu; size %lu", manifest_location, manifest_size);
}

}  // namespace intel_npu
