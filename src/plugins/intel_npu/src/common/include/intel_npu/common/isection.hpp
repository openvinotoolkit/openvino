// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

// TODOs: fix the circular dependencies
// Move sections in directory
// Unique SID - how do we reinforce this without compromising modularity? Description matching?

using SectionID = uint16_t;
using BlobSource = std::variant<std::reference_wrapper<std::istream>,
                                std::reference_wrapper<const ov::Tensor>>;  // TODO find a better place

class BlobWriter;
class BlobReader;

namespace PredefinedSectionID {
enum {
    CRE = 100,
    OFFSETS_TABLE = 101,
    ELF_MAIN_SCHEDULE = 102,
    ELF_INIT_SCHEDULES = 103,
    IO_LAYOUTS = 104,
    BATCH_SIZE = 105,
};
};

class ISection {
public:
    ISection(const SectionID section_id);

    virtual ~ISection() = default;

    virtual void write(std::ostream& stream, BlobWriter* writer) = 0;

    // note necessary, saves some performance if provided
    virtual std::optional<uint64_t> get_length() const;

    SectionID get_section_id() const;

private:
    SectionID m_section_id;
};

}  // namespace intel_npu
