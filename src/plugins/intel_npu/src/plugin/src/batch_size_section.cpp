// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace intel_npu {

BatchSizeSection::BatchSizeSection(const int64_t batch_size)
    : ISection(PredefinedSectionType::BATCH_SIZE),
      m_batch_size(batch_size) {}

void BatchSizeSection::write(std::ostream& stream, BlobWriter* writer) {
    stream.write(reinterpret_cast<const char*>(&m_batch_size), sizeof(m_batch_size));
}

int64_t BatchSizeSection::get_batch_size() const {
    return m_batch_size;
}

std::shared_ptr<ISection> BatchSizeSection::read(BlobReader* blob_reader, const size_t section_length) {
    OPENVINO_ASSERT(section_length == sizeof(int64_t),
                    "BatchSizeSection: incorrect section length ",
                    section_length,
                    ". Expected: ",
                    sizeof(int64_t));

    int64_t batch_size;
    blob_reader->copy_data_from_source(reinterpret_cast<char*>(&batch_size), sizeof(batch_size));

    return std::make_shared<BatchSizeSection>(batch_size);
}

}  // namespace intel_npu
