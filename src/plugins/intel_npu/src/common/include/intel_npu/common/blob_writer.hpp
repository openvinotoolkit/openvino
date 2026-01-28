// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "cre.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

class BlobWriter {
public:
    BlobWriter();

    BlobWriter(const std::shared_ptr<BlobReader>& blob_reader);

    void register_section(const std::shared_ptr<ISection>& section);

    void register_offset_in_table(const SectionID id, const uint64_t offset);

    void write(std::ostream& stream);

    void append_compatibility_requirement(const CRE::Token requirement_token);

    void append_compatibility_requirement(const std::vector<CRE::Token>& requirement_tokens);

    std::streamoff get_stream_relative_position(std::ostream& stream) const;

private:
    void write_section(std::ostream& stream, const std::shared_ptr<ISection>& section);

    std::unordered_set<SectionID> m_registered_sections_ids;
    std::queue<std::shared_ptr<ISection>> m_registered_sections;
    CRE m_cre;
    std::unordered_map<SectionID, uint64_t> m_offsets_table;

    /**
     * @brief TODO
     */
    std::optional<std::streampos> m_stream_base = std::nullopt;

    Logger m_logger;
};

}  // namespace intel_npu
