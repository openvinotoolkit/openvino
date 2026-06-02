// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"
#include "intel_npu/common/itt.hpp"

namespace intel_npu {

BatchSizeSection::BatchSizeSection(const int64_t batch_size)
    : ISection(PredefinedSectionType::BATCH_SIZE),
      m_batch_size(batch_size),
      m_logger("BatchSizeSection", Logger::global().level()) {
    m_logger.trace("Section created");
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

    Logger("BatchSizeSection", Logger::global().level()).debug("Read batch size %lu", batch_size);

    return std::make_shared<BatchSizeSection>(batch_size);
}

}  // namespace intel_npu
