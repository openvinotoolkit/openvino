// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_size_section.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace {

constexpr intel_npu::ISection::SectionID BATCH_SIZE_SECTION_ID = 105;

}  // namespace

namespace intel_npu {

BatchSizeSection::BatchSizeSection(const int64_t batch_size)
    : ISection(BATCH_SIZE_SECTION_ID),
      m_batch_size(batch_size) {}

void BatchSizeSection::write(std::ostream& stream, BlobWriter* writer) {
    stream.write(reinterpret_cast<const char*>(&m_batch_size), sizeof(m_batch_size));
    writer->cursor +=
        sizeof(m_batch_size);  // TODO maybe the cursor should be moved "automatically" to the end after this call
}

std::optional<uint64_t> BatchSizeSection::get_length() const {
    return sizeof(m_batch_size);
}

}  // namespace intel_npu
