// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace intel_npu {

// TODOs: fix the circular dependencies
// Move sections in directory
// Create initschedules & main schedule sections. these hold a graph object, call the right "export"
// Past capabilities
// Unique SID

class BlobWriter;
class BlobReader;

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
