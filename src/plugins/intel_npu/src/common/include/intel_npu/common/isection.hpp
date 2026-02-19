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

/**
 * @brief Identifies the type of the section, along with its corresponding read & write handlers.
 */
using SectionType = uint16_t;
/**
 * @brief Used to distinguish multiple sections of the same type within the same compiled model.
 */
using SectionTypeInstance = uint16_t;

/**
 * @brief Uniquely identifies a section within a compiled model.
 */
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

/**
 * @brief Section types already known by the NPU plugin. These section type IDs are reserved.
 */
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

/**
 * @brief Interface that should be implemented by all blob section handlers. Its role is to standardize the
 * identification of the section, along with the signatures of the read & writer handlers.
 */
class ISection {
public:
    ISection(const SectionType type);

    virtual ~ISection() = default;

    /**
     * @brief Method used to instruct the BlobWriter how to write the current section into the provided stream.
     */
    virtual void write(std::ostream& stream, BlobWriter* writer) = 0;

    SectionType get_section_type() const;

    /**
     * @brief Get the type instance ID
     *
     * @return Either the ID or a std::nullopt. This value exists only if the current section has been added to a
     * BlobWriter writing queue.
     */
    std::optional<SectionTypeInstance> get_section_type_instance() const;

    /**
     * @brief Get the section ID, unique per compiled model.
     *
     * @return Either the ID or a std::nullopt. This value exists only if the current section has been added to a
     * BlobWriter writing queue.
     */
    std::optional<SectionID> get_section_id() const;

private:
    friend class BlobWriter;
    friend class BlobReader;

    /**
     * @brief Standard setter.
     * @note Only BlobWriters & BlobReaders should be allowed to manipulate the type instance ID.
     */
    void set_section_type_instance(const SectionTypeInstance type_instance) const;

    SectionType m_section_type;
    /**
     * @note This value exists only if the current section has been added to a BlobWriter writing queue.
     */
    mutable std::optional<SectionTypeInstance> m_section_type_instance;
};

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionID> {
    size_t operator()(const intel_npu::SectionID& sid) const;
};
