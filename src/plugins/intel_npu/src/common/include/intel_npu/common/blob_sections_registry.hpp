// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "intel_npu/common/isection.hpp"
#include "intel_npu/common/runtime_requirements.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

class BlobSectionsRegistry final {
public:
    BlobSectionsRegistry();

    /**
     * @brief Add a new blob section to the writing queue.
     *
     * @param section The section to be added in the writing queue.
     * @param id TODO
     * @return The instance ID of the section. This number corresponds to the order within the writing queue and it is
     * used to distringuish between sections of the same type. This number is unique only among sections of the same
     * type.
     */
    SectionID register_section(const std::shared_ptr<ISection>& section,
                               const std::optional<SectionID>& parsed_id = std::nullopt);

    /**
     * @brief Retrieve a parsed section.
     * @note This should be called only after "read" was invoked.
     * @return The requested section if found, else "nullptr"
     */
    std::shared_ptr<ISection> retrieve_section(const SectionID& id) const;

    /**
     * @brief Retrieve the first parsed section of the given type.
     * @note This should be called only after "read" was invoked.
     * @note This function exists only for convenience. Most section types will typically have a single instance inside
     * a compiled model.
     * @return The first parsed section of the given type if any, else "nullptr"
     */
    std::shared_ptr<ISection> retrieve_any_section(const SectionType section_type) const;

    /**
     * @brief Retrieves all parsed sections of the given type.
     * @note This should be called only after "read" was invoked.
     * @return All parsed sections of the given type if any, else "std::nullopt"
     */
    std::vector<std::shared_ptr<ISection>> retrieve_sections(const SectionType type) const;

    std::queue<std::shared_ptr<ISection>> get_write_queue() const;

    /**
     * @return true if at least one section of the given type has been parsed
     * @return false otherwise
     */
    bool has_section_of_type(const SectionType section_type) const;

    /**
     * @return The number of parsed sections of the given type
     */
    size_t count_sections_of_type(const SectionType section_type) const;

    /**
     * @return A summary of the blob content, represented as a map between section types and their number of occurrences
     * in the blob
     */
    std::unordered_map<SectionType, size_t> get_content_summary() const;

    // TODO update
    /**
     * @brief Builds the runtime requirements (CRE + individual "compatibility substrings") based on the sections
     * registered so far.
     * @note The requirements are built on demand and not stored in order to make sure the requirements are aligned with
     * the content of the blob.
     */
    std::shared_ptr<ISection> build_runtime_requirements_section() const;

private:
    /**
     * @brief Tracks the next available instance ID for each section type. This should assure that the generated section
     * IDs (type + instance) are unique per compiled model.
     */
    uint16_t m_next_section_id;
    /**
     * @brief Queue that holds all sections to be written at export time.
     */
    // TODO consider storing section ID in all attrs but m_ids_to_sections
    std::queue<std::shared_ptr<ISection>> m_write_queue;
    std::unordered_map<SectionID, std::shared_ptr<ISection>> m_ids_to_sections;
    std::unordered_map<SectionType, std::vector<std::shared_ptr<ISection>>> m_types_to_sections;

    Logger m_logger;
};

}  // namespace intel_npu
