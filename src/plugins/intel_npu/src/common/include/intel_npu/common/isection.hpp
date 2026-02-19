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

#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

// TODOs: fix the circular dependencies
// Consider moving the secion files in dedicated directories

using SectionType = uint16_t;
using SectionTypeInstance = uint16_t;

struct SectionID final {
    SectionID() = default;
    SectionID(SectionType section_type, SectionTypeInstance section_type_instance);

    SectionType type;
    SectionTypeInstance type_instance;
};

bool operator==(const SectionID& sid1, const SectionID& sid2);

std::ostream& operator<<(std::ostream& out, const SectionID& id);

class BlobWriter;
class BlobReader;

namespace PredefinedSectionType {
enum {
    CRE = 100,
    OFFSETS_TABLE = 101,
    ELF_MAIN_SCHEDULE = 102,
    ELF_INIT_SCHEDULES = 103,
    IO_LAYOUTS = 104,
    BATCH_SIZE = 105,
};
};

// Only a single instance should exist within a blob. So, we may predefine these IDs for convenience.
const SectionID CRE_SECTION_ID(PredefinedSectionType::CRE, 0);
const SectionID OFFSETS_TABLE_SECTION_ID(PredefinedSectionType::OFFSETS_TABLE, 0);

class ISection {
public:
    ISection(const SectionType type);

    virtual ~ISection() = default;

    virtual void write(std::ostream& stream, BlobWriter* writer) = 0;

    SectionType get_section_type() const;

    std::optional<SectionTypeInstance> get_section_type_instance() const;

    std::optional<SectionID> get_section_id() const;

private:
    friend class BlobWriter;
    friend class BlobReader;

    void set_section_type_instance(const SectionTypeInstance type_instance) const;

    SectionType m_section_type;
    mutable std::optional<SectionTypeInstance> m_section_type_instance;
};

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionID> {
    size_t operator()(const intel_npu::SectionID& sid) const;
};
