// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

#include "intel_npu/common/isection.hpp"

namespace intel_npu {

class OffsetsTableSection final : public ISection {
public:
    OffsetsTableSection(const std::unordered_map<ISection::SectionID, uint64_t>& offsets_table);

    void write(std::ostream& stream, BlobWriter* writer) override;

    std::optional<uint64_t> get_length() const override;

private:
    std::reference_wrapper<const std::unordered_map<ISection::SectionID, uint64_t>> m_offsets_table;
};

}  // namespace intel_npu
