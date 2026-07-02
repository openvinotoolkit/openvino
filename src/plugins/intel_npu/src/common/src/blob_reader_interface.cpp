// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/blob_reader_interface.hpp"

namespace intel_npu {

BlobReaderInterface::BlobReaderInterface(const ov::Tensor& source,
                                         const size_t section_start,
                                         const size_t section_length,
                                         const size_t npu_region_size,
                                         const ov::log::Level log_level)
    : m_source(source),
      m_cursor(section_start),
      m_section_start(section_start),
      m_section_end(section_start + section_length),
      m_logger("BlobReaderInterface", log_level) {
    OPENVINO_ASSERT(m_section_end <= npu_region_size,
                    "The end of a section surpasses the registered NPU region boundaries. Section end position: ",
                    m_section_end,
                    ". Limit: ",
                    npu_region_size);
    m_logger.debug("Created a new BlobReaderInterface. Boundaries: [%lu, %lu)", m_section_start, m_section_end);
}

void BlobReaderInterface::copy_from_source(char* destination, const size_t size) {
    m_logger.trace("Reading and copying %lu bytes", size);

    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_section_end, "A section reader attempted to read beyond its own boundaries");
    std::memcpy(destination, m_source.get().data<const char>() + m_cursor - size, size);
}

const void* BlobReaderInterface::interpret_from_source(const size_t size) {
    m_logger.trace("Reading without copying %lu bytes", size);

    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_section_end, "A section reader attempted to read beyond its own boundaries");
    return reinterpret_cast<const void*>(m_source.get().data<char>() + m_cursor - size);
}

ov::Tensor BlobReaderInterface::get_roi_tensor(const size_t size) {
    m_logger.trace("Extracting an RoI tensor of %lu bytes", size);

    m_cursor += size;
    OPENVINO_ASSERT(m_cursor <= m_section_end, "A section reader attempted to read beyond its own boundaries");
    return ov::Tensor(m_source, ov::Coordinate{m_cursor - size}, ov::Coordinate{m_cursor});
}

size_t BlobReaderInterface::get_offset_relative_to_current_section() const {
    return m_cursor - m_section_start;
}

void BlobReaderInterface::move_cursor_relative_to_current_section(const size_t offset) {
    m_cursor = m_section_start + offset;
    OPENVINO_ASSERT(m_cursor <= m_section_end,
                    "A section reader attempted to move the cursor beyond its own boundaries");
}

size_t BlobReaderInterface::get_offset_relative_to_npu_region() const {
    return m_cursor;
}

void BlobReaderInterface::move_cursor_relative_to_npu_region(const size_t offset) {
    OPENVINO_ASSERT(offset >= m_section_start && offset <= m_section_end,
                    "A section reader attempted to move the cursor beyond its own boundaries");
    m_cursor = offset;
}

size_t BlobReaderInterface::get_section_length() const {
    return m_section_end - m_section_start;
}

ov::log::Level BlobReaderInterface::get_log_level() const {
    return m_logger.level();
}

}  // namespace intel_npu
