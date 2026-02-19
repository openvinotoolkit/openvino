// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "cre.hpp"
#include "intel_npu/common/offsets_table.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

/**
 * @brief Class responsible for exporting a compiled model.
 * @details There should be a 1:1 mapping between "CompiledModel" & "BlobWriter" instances.
 *
 * When the user requests the compilation of a model, a "BlobWriter" object is created. During the compilation flow,
 * this object gathers the blob sections that should be written later if the user requests an export. These sections are
 * actually classes implementing the "ISection" interface. Their role is to instruct the BlobWriter how to fill the blob
 * if an exportation is requested.
 *
 * The BlobWriter is also holding the compatibility requirements expression (CRE) and exposes an API that allows filling
 * it.
 *
 * Additionally, the BlobWriter exposes an API required to meet the needs of all custom section writers (implemented in
 * the class inheriting "ISection").
 */
class BlobWriter {
public:
    BlobWriter();

    /**
     * @brief Construct a BlobWriter based on the data parse by the BlobReader.
     * @details The BlobReader handles the parsing of a compiled model. The data parsed by this object can be reused to
     * re-export the model using a BlobWriter if requested.
     *
     * @param blob_reader Contains the parsed data of a compiled model.
     */
    BlobWriter(const std::shared_ptr<BlobReader>& blob_reader);

    /**
     * @brief Add a new blob section to the writing queue.
     *
     * @param section The section to be added in the writing queue.
     * @return The instance ID of the section. This number corresponds to the order within the writing queue and it is
     * used to distringuish between sections of the same type. This number is unique only among sections of the same
     * type.
     */
    SectionTypeInstance register_section(const std::shared_ptr<ISection>& section);

    /**
     * @brief Writes all sections within the writing queue into the provided stream.
     * @note This operation is idempotent. I.e. calling this function twice in a row (but on different streams) will
     * yield tha same result. This is done by saving & restoring the attributes of the class, the writing queue
     * included.
     *
     * @param stream Where the blob will be stored.
     */
    void write(std::ostream& stream);

    /**
     * @brief Append a new token to the CRE, at depth-level 1. All tokens found at this depth-level are bound by a
     * logical "AND" operator.
     *
     * @param requirement_token
     */
    void append_compatibility_requirement(const CRE::Token requirement_token);

    /**
     * @brief Append a new CRE subexpression to the CRE, at depth-level 1. All tokens found at this depth-level are
     * bound by a logical "AND" operator.
     *
     * @param requirement_token
     */
    void append_compatibility_requirement(const std::vector<CRE::Token>& requirement_tokens);

    /**
     * @brief Get the position of the write cursor relative to the beginning of the NPU region.
     */
    std::streamoff get_stream_relative_position(std::ostream& stream) const;

    /**
     * @brief Move the position of the write cursor.
     * @note This operation allows moving the cursor only within the region allocated to the indicated section. The
     * parsing of each blob section should be done indepent of other sections, thus this constraint.
     *
     * @param stream The target stream.
     * @param section_id Used to check if the offset falls within the region allocated to this section.
     * @param offset Where to move the cursor. This number is relative to the beginning of the NPU region.
     */
    void move_stream_cursor_to_relative_position(std::ostream& stream,
                                                 const SectionID section_id,
                                                 const uint64_t offset);

private:
    /**
     * @brief Helper function. Registers a section that has been already parsed by the BlobReader.
     */
    void register_section_from_blob_reader(const std::shared_ptr<ISection>& section);

    /**
     * @brief Helper function that writes the given section to the stream.
     * @details Calls the "write" method of the given section to fill the payload. Then adds a new entry inside the
     * table of offsets.
     */
    void write_section(std::ostream& stream, const std::shared_ptr<ISection>& section);

    /**
     * @brief Tracks the next available instance ID for each section type. This should assure that the generated section
     * IDs (type + instance) are unique per compiled model.
     */
    std::unordered_map<SectionType, SectionTypeInstance> m_next_type_instance_id;
    /**
     * @brief Queue that holds all sections to be written at export time.
     */
    std::queue<std::shared_ptr<ISection>> m_registered_sections;
    CRE m_cre;
    OffsetsTable m_offsets_table;

    /**
     * @brief Holds the offset where the NPU blob region begins. This attribute has value only during a "write"
     * operation.
     */
    std::optional<std::streampos> m_stream_base = std::nullopt;

    Logger m_logger;
};

}  // namespace intel_npu
