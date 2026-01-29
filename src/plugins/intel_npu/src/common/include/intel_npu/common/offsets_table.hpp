// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class OffsetsTable final {
public:
    OffsetsTable() = default;

    void add_entry(const SectionID id, const uint64_t offset, const uint64_t length);

    static size_t get_entry_size();

    std::optional<uint64_t> lookup_offset(const SectionID id) const;

    std::optional<uint64_t> lookup_length(const SectionID id) const;

    std::optional<SectionID> lookup_section_id(const uint64_t offset) const;

private:
    friend class OffsetsTableSection;

    std::unordered_map<SectionID, std::pair<uint64_t, uint64_t>> m_table;
    std::unordered_map<uint64_t, SectionID> m_reversed_table;
};

class OffsetsTableSection final : public ISection {
public:
    OffsetsTableSection(const OffsetsTable& offsets_table);

    void write(std::ostream& stream, BlobWriter* writer) override;

    OffsetsTable get_table() const;

    static std::shared_ptr<ISection> read(BlobReader* blob_reader, const size_t section_length);

private:
    OffsetsTable m_offsets_table;
};

}  // namespace intel_npu
