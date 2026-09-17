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

#include "intel_npu/common/cre_token.hpp"
#include "intel_npu/common/section_id.hpp"
#include "intel_npu/common/section_type.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

// TODOs: fix the circular dependencies
// Consider moving the secion files in dedicated directories

/**
 * @brief Converts the given section type and ID pair to a string.
 * @return A string of the form <section type>_<section ID>, where the section ID is an integer.
 */
std::string section_type_and_id_to_string(const SectionType type, const SectionID id);

/**
 * @brief Parses a section type from string. If a section ID is also found at the end of the string, then the ID is
 * returned as well.
 * @details If an ID is present, then the expected form of the string content is <section type>_<section ID>, where the
 * section ID is an integer.
 */
std::pair<SectionType, std::optional<SectionID>> section_type_and_id_from_string(std::string_view type_and_id);

class BlobWriterInterface;
class BlobReaderInterface;

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

    SectionType get_type() const;

    /**
     * @brief Get the section ID, unique per compiled model.
     *
     * @return Either the ID or a std::nullopt. This value exists only if the current section has been added to a
     * BlobWriter writing queue.
     */
    std::optional<SectionID> get_id() const;

    // TODO rename?
    /**
     * @brief Builds and returns the runtime requirements of the current section instance as a string.
     * @note The returned string should follow the "value" format dictated by the "compatibility string" parser:
     * value ::= [A-Z0-9][_A-Z0-9\.]*
     *
     * @return The requirements as a string if any. std::nullopt otherwise.
     */
    virtual std::optional<std::string> get_individual_compatibility_requirements() const;

    /**
     * @brief Get the compatibility requirements subexpression corresponding to the current section.
     * @details The base implementation doesn't return any requirements. If there are any compatibility requirements,
     * these would typically be <section type> (e.g. ELF_INIT_SCHEDULE) or <section type> + <section ID> (e.g.
     * ELF_INIT_SCHEDULE_1). This method can be overriden correspondingly.
     *
     * More complex expressions, that take into account some other registered sections can also be returned. For
     * example, if we wish to register something like "ELF_INIT_SCHEDULE_1 OR ELF_INIT_SCHEDULE_2", then we may override
     * this function to have the section of the first schedule write the OR relationship. The other section could then
     * write nothing (to avoid redundancy).
     * @note The subexpression returned by this function will be stitched to the main CRE using a logical "AND".
     * @param all_registered_sections A map offering access to all sections registered for the current writing
     * section. Relevant if the requirements should take other sections into account.
     * @return The subexpression describing the requirements of the current section.
     */
    virtual std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const;

private:
    // Access required to set the section type instance ID
    friend class BlobWriter;
    friend class BlobWriterInterface;
    friend class BlobReader;

    /**
     * @note Only BlobWriters & BlobReaders should be allowed to manipulate the section ID. This is because the
     * instance ID denotes, by convention, the order in which the sections of the given type have been registered within
     * the writing queue.
     */
    void set_id(const SectionID& id) const;

    SectionType m_type;
    /**
     * @note This value exists only if the current section has been added to a BlobWriter writing queue.
     */
    mutable std::optional<SectionID> m_id;
};

}  // namespace intel_npu
