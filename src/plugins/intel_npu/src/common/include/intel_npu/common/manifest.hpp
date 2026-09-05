// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

/**
 * @brief Manifest meant to be integrated within the NPU blob format.
 * @note Although this implementation is used by the main manifest section of the NPU blob region, it can be reused
 * for use cases within the payload of other sections.
 */
class Manifest final {
public:
    Manifest(const ov::log::Level log_level = ov::log::Level::WARNING);

    void add_entry(const SectionID id, const SectionType type, const uint64_t offset, const uint64_t length);

    static size_t get_entry_size();

    std::optional<SectionType> lookup_type(const SectionID id) const;

    std::optional<uint64_t> lookup_offset(const SectionID id) const;

    std::optional<uint64_t> lookup_length(const SectionID id) const;

    std::optional<SectionID> lookup_section_id(const uint64_t offset) const;

    size_t get_number_of_entries() const;

    std::unordered_set<SectionID> get_all_registered_section_ids() const;

    bool empty() const;

private:
    friend class ManifestSection;

    /**
     * @brief From section IDs to section types, offsets & lengths.
     */
    std::unordered_map<SectionID, std::tuple<SectionType, uint64_t, uint64_t>> m_table;
    /**
     * @brief From offsets to section IDs.
     */
    std::unordered_map<uint64_t, SectionID> m_reversed_table;

    Logger m_logger;
};

class ManifestSection final : public ISection {
public:
    ManifestSection(const Manifest& manifest, const ov::log::Level log_level = ov::log::Level::WARNING);

    void write(BlobWriterInterface& writer) override;

    Manifest get_table() const;

    static std::shared_ptr<ISection> read(BlobReaderInterface& blob_reader);

private:
    Manifest m_manifest;

    Logger m_logger;
};

}  // namespace intel_npu
