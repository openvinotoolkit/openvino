// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Table of offsets meant to be integrated within the NPU blob format.
 * @note Although this implementation is used by the main offsets table section of the NPU blob region, it can be reused
 * for use cases within the payload of other sections.
 */
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

    /**
     * @brief From section IDs to offsets & lengths.
     */
    std::unordered_map<SectionID, std::pair<uint64_t, uint64_t>> m_table;
    /**
     * @brief From offsets to section IDs.
     */
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
