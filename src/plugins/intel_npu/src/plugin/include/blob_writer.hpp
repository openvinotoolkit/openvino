// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cre.hpp"
#include "intel_npu/common/igraph.hpp"
#include "isection.hpp"

namespace intel_npu {

class BlobWriter {
public:
    BlobWriter();

    void register_section(const std::shared_ptr<ISection>& section);

    void register_offset_in_table(const ISection::SectionID id, const uint64_t offset);

    void write(std::ostream& stream, const std::shared_ptr<IGraph>& graph);

    void append_compatibility_requirement(const CREToken requirement_token);

    size_t offset = 0;

private:
    void write_persistent_format_region();

    std::unordered_set<ISection::SectionID> m_registered_sections_ids;
    std::vector<std::shared_ptr<ISection>> m_registered_sections;
    std::unordered_map<ISection::SectionID, uint64_t> m_offsets_table;
    std::shared_ptr<CRESection> m_cre;
};

}  // namespace intel_npu
