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

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

// TODOs: fix the circular dependencies
// Consider moving the secion files in dedicated directories

using CREToken = uint16_t;

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

    std::string to_string() const;

    static SectionID from_string(std::string_view section_id_string);

    SectionType type;
    SectionTypeInstance type_instance;
};

bool operator==(const SectionID& sid1, const SectionID& sid2);

std::ostream& operator<<(std::ostream& os, const SectionID& id);

std::istream& operator>>(std::istream& is, SectionID& id);

class BlobWriterInterface;
class BlobReaderInterface;

/**
 * @brief Section types already known by the NPU plugin. These section type IDs are reserved.
 */
namespace PredefinedSectionType {
enum : SectionType {
    CRE = 100,
    OFFSETS_TABLE = 101,
    ELF_MAIN_SCHEDULE = 102,
    ELF_INIT_SCHEDULES = 103,
    IO_LAYOUTS = 104,
    BATCH_SIZE = 105,
};
};

// TODO any point in adding the offsets table as well?
static inline const std::unordered_set<SectionType> DEFAULT_SUPPORTED_SECTION_TYPES{
    PredefinedSectionType::CRE,
    PredefinedSectionType::ELF_MAIN_SCHEDULE,
    PredefinedSectionType::ELF_INIT_SCHEDULES,
    PredefinedSectionType::BATCH_SIZE};

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
    virtual void write(BlobWriterInterface& writer) = 0;

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

    /**
     * @brief Get the compatibility requirements subexpression corresponding to the current section.
     * @details The base implementation returns the section ID (type ID + type instance ID) as the required
     * subexpression. This implementation can be overriden to take into consideration other registered sections as well.
     *
     * For example, if we wish to register something like "ELF_INIT_SCHEDULE_1 OR ELF_INIT_SCHEDULE_2", then we may
     * override this function to have the section of the first schedule write the OR relationship. The other section
     * could then write nothing.
     * @note The subexpression returned by this function is meant to be stitched to the main CRE using a logical "AND".
     * @param all_registered_sections A map offering access to all sections registered for the current writing section.
     * @return The subexpression describing the requirements of the current section.
     */
    virtual std::vector<CREToken> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>&
            all_registered_sections) const;

    /**
     * @brief Evaluate whether or not the current section instance is compatible with the current environment based on
     * the content of the section.
     * @details The first step in determining the compatibility of a section is by evaluating the compatibility of its
     * type (see TODO). The second step is this function, which evaluates the compatibility of the current instance.
     *
     * The section writers are able to handle additional compatibility requirements by using the content of their own
     * section. This function is meant to evaluate the said content if the case is applicable.
     * @param reader The blob content of the section, as well as the capabilities of the plugin are available through
     * this object.
     */
    virtual bool evaluate_compatibility_based_on_section_content(BlobReaderInterface& reader);

private:
    // Access required to set the section type instance ID
    friend class BlobWriter;
    friend class BlobWriterInterface;
    friend class BlobReader;

    /**
     * @note Only BlobWriters & BlobReaders should be allowed to manipulate the type instance ID. This is because the
     * instance ID denotes, by convention, the order in which the sections of the given type have been registered to be
     * written in the blob.
     */
    void set_section_type_instance(const SectionTypeInstance type_instance) const;

    SectionType m_section_type;
    /**
     * @note This value exists only if the current section has been added to a BlobWriter writing queue.
     */
    mutable std::optional<SectionTypeInstance> m_section_type_instance;

    /**
     * @brief Stores the result obtained after evaluating if the current section is supported based on its section
     * content. This result can then be returned in future calls, thus avoiding the need to reevaluate.
     */
    mutable std::optional<bool> m_type_instance_supported;
};

}  // namespace intel_npu

template <>
struct std::hash<intel_npu::SectionID> {
    size_t operator()(const intel_npu::SectionID& sid) const;
};
