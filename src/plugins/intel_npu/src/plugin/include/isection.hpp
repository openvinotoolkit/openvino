// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "blob_reader.hpp"
#include "blob_writer.hpp"

namespace intel_npu {

class ISection {
public:
    using SectionID = uint16_t;

    ISection(const SectionID section_id);

    virtual void write(std::ostream& stream, BlobWriter* writer) = 0;

    // note necessary, saves some performance if provided
    virtual std::optional<uint64_t> get_length() const;

    // virtual void read(BlobReader* reader) = 0;

    SectionID get_section_id() const;

private:
    SectionID m_section_id;
};

}  // namespace intel_npu
