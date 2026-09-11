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

std::string section_type_and_id_to_string(const SectionType type, const SectionID id);

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
    // TODO add note to respect the format:
    // value ::= [A-Z0-9][_A-Z0-9\.]*
    virtual std::optional<std::string> get_individual_compatibility_requirements() const;

    /**
     * @brief Get the compatibility requirements subexpression corresponding to the current section.
     * @details The base implementation returns the section ID (type ID + type instance ID) as the required
     * subexpression. This implementation can be overriden to take into consideration other registered sections as
     * well.
     *
     * For example, if we wish to register something like "ELF_INIT_SCHEDULE_1 OR ELF_INIT_SCHEDULE_2", then we may
     * override this function to have the section of the first schedule write the OR relationship. The other section
     * could then write nothing.
     * @note The subexpression returned by this function is meant to be stitched to the main CRE using a logical
     * "AND".
     * @param all_registered_sections A map offering access to all sections registered for the current writing
     * section.
     * @return The subexpression describing the requirements of the current section.
     */
    virtual std::vector<std::shared_ptr<CREToken>> get_compatibility_requirements_subexpression(
        const std::unordered_map<SectionID, std::shared_ptr<ISection>>& all_registered_sections) const;

    /**
     * @brief Evaluate whether or not the current section instance is compatible with the current environment based on
     * the content of the section.
     * @details The first step in determining the compatibility of a section is by evaluating the compatibility of its
     * type. The second step is this function, which evaluates the compatibility of the current instance.
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
    void set_id(const SectionID& id) const;

    SectionType m_type;
    /**
     * @note This value exists only if the current section has been added to a BlobWriter writing queue.
     */
    mutable std::optional<SectionID> m_id;
};

}  // namespace intel_npu
