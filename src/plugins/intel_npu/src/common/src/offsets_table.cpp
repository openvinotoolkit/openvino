// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/offsets_table.hpp"

#include "intel_npu/common/blob_reader.hpp"
#include "intel_npu/common/blob_writer.hpp"

namespace {

constexpr intel_npu::ISection::SectionID OFFSETS_TABLE_SECTION_ID = 101;

}  // namespace

namespace intel_npu {

OffsetsTableSection::OffsetsTableSection(const std::unordered_map<ISection::SectionID, uint64_t>& offsets_table)
    : ISection(OFFSETS_TABLE_SECTION_ID),
      m_offsets_table(offsets_table) {}

void OffsetsTableSection::write(std::ostream& stream, BlobWriter* writer) {
    for (const auto& [key, value] : m_offsets_table.get()) {
        stream.write(reinterpret_cast<const char*>(key), sizeof(key));
        stream.write(reinterpret_cast<const char*>(value), sizeof(value));
    }
}

std::optional<uint64_t> OffsetsTableSection::get_length() const {
    return m_offsets_table.get().size() * (sizeof(ISection::SectionID) + sizeof(uint64_t));
}

}  // namespace intel_npu
