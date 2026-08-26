// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader_interface.hpp"

#include "intel_npu/config/options.hpp"

namespace intel_npu {

BlobReaderInterface::BlobReaderInterface(BlobSource& source,
                                         const size_t npu_region_start,
                                         const size_t npu_region_size,
                                         const size_t section_start,
                                         const size_t section_length,
                                         const FilteredConfig& config)
    : m_source(source),
      m_npu_region_start(npu_region_start),
      m_section_start(section_start),
      m_section_end(section_start + section_length),
      m_config(config),
      m_logger("BlobReaderInterface", config.get<LOG_LEVEL>()) {
    OPENVINO_ASSERT(section_start <= m_section_end, "Integer overflow while computing the end boundary of a section");
    OPENVINO_ASSERT(npu_region_start <= npu_region_start + npu_region_size,
                    "Integer overflow while computing the end boundary of the NPU blob region");
    OPENVINO_ASSERT(npu_region_start <= section_start,
                    m_section_end <= npu_region_start + npu_region_size,
                    "The given limits of the section do not fit within the limits of the NPU blob region");
    m_logger.debug("Created a new BlobReaderInterface. Boundaries: [%lu, %lu)", m_section_start, m_section_end);
}

void BlobReaderInterface::read_into_buffer(void* destination, const size_t size) {
    m_logger.trace("Reading and copying %lu bytes", size);

    OPENVINO_ASSERT(size <= m_section_end && m_source.get().tellg() <= m_section_end - size,
                    "A section reader attempted to read beyond its own boundaries");
    m_source.get().read_into_buffer(destination, size);
}

const void* BlobReaderInterface::read_view(const size_t size) {
    m_logger.trace("Reading without copying %lu bytes", size);

    OPENVINO_ASSERT(size <= m_section_end && m_source.get().tellg() <= m_section_end - size,
                    "A section reader attempted to read beyond its own boundaries");
    return m_source.get().read_view(size);
}

ov::Tensor BlobReaderInterface::create_roi_tensor(const size_t size) {
    m_logger.trace("Extracting an RoI tensor of %lu bytes", size);

    OPENVINO_ASSERT(size <= m_section_end && m_source.get().tellg() <= m_section_end - size,
                    "A section reader attempted to read beyond its own boundaries");
    return m_source.get().create_roi_tensor(size);
}

size_t BlobReaderInterface::get_offset_relative_to_current_section() const {
    OPENVINO_ASSERT(m_source.get().tellg() >= m_section_start, "Invalid cursor position");
    return m_source.get().tellg() - m_section_start;
}

// TODO rename these
void BlobReaderInterface::move_cursor_relative_to_current_section(const size_t offset) {
    OPENVINO_ASSERT(m_section_start <= m_section_start + offset && m_section_start + offset <= m_section_end,
                    "A section reader attempted to move the cursor beyond its own boundaries");
    m_source.get().seekg(m_section_start + offset, std::ios::beg);
}

size_t BlobReaderInterface::get_offset_relative_to_npu_region() const {
    OPENVINO_ASSERT(m_source.get().tellg() >= m_section_start, "Invalid cursor position");
    return m_source.get().tellg() - m_npu_region_start;
}

void BlobReaderInterface::move_cursor_relative_to_npu_region(const size_t offset) {
    const size_t destination = m_npu_region_start + offset;
    OPENVINO_ASSERT(m_npu_region_start <= destination, "Integer overflow");

    OPENVINO_ASSERT(destination >= m_section_start && destination <= m_section_end,
                    "A section reader attempted to move the cursor beyond its own boundaries");
    m_source.get().seekg(destination, std::ios::beg);
}

bool BlobReaderInterface::source_is_contiguous() const {
    return m_source.get().is_contiguous();
}

size_t BlobReaderInterface::get_section_length() const {
    return m_section_end - m_section_start;
}

FilteredConfig BlobReaderInterface::get_config() const {
    return m_config;
}

ov::log::Level BlobReaderInterface::get_log_level() const {
    return m_logger.level();
}

}  // namespace intel_npu
