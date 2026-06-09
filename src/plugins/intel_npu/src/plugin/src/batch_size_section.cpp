// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

BatchSizeSection::BatchSizeSection(const int64_t batch_size, const ov::log::Level log_level)
    : ISection(PredefinedSectionType::BATCH_SIZE),
      m_batch_size(batch_size),
      m_logger("BatchSizeSection", log_level) {
    m_logger.trace("Section created");
}

std::vector<CREToken> BatchSizeSection::get_compatibility_requirements_subexpression(
    const std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>&
    /*all_registered_sections*/) const {
    m_logger.debug("Added the BATCH_SIZE section type to the CRE");
    return {get_section_type()};
}

void BatchSizeSection::write(BlobWriterInterface& writer) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BatchSizeSection::write");
    m_logger.debug("Writting batch size %lu", m_batch_size);

    writer.write(&m_batch_size, sizeof(m_batch_size));
}

int64_t BatchSizeSection::get_batch_size() const {
    return m_batch_size;
}

std::shared_ptr<ISection> BatchSizeSection::read(BlobReaderInterface& blob_reader) {
    OV_ITT_SCOPED_TASK(itt::domains::NPUPlugin, "BatchSizeSection::read");

    const size_t section_length = blob_reader.get_section_length();
    OPENVINO_ASSERT(section_length == sizeof(int64_t),
                    "BatchSizeSection: incorrect section length ",
                    section_length,
                    ". Expected: ",
                    sizeof(int64_t));

    int64_t batch_size;
    blob_reader.copy_data_from_source(reinterpret_cast<char*>(&batch_size), sizeof(batch_size));

    Logger("BatchSizeSection", blob_reader.get_log_level()).debug("Read batch size %lu", batch_size);

    return std::make_shared<BatchSizeSection>(batch_size, blob_reader.get_log_level());
}

}  // namespace intel_npu
