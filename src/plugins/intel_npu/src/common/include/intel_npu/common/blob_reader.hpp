// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "intel_npu/common/icapability.hpp"
#include "intel_npu/common/offsets_table.hpp"

namespace intel_npu {

/**
 * @brief Class responsible for parsing the NPU specific data of a compiled model.
 * @details There should be a 1:1 mapping between "CompiledModel" & "BlobReader" instances.
 *
 * When the user requests the importation of a model, a "BlobReader" object is created. All known section readers will
 * be registered into this object. Later these readers will be used to parse individual sections residing within the
 * compiled model. The parsed sections can then be retrieved and used by the NPU plugin.
 *
 * During the parse procedure, a compatibility requirements expression (CRE) is evaluated as one of the first steps. If
 * the evaluation yields a negative result, then the current version of the plugin cannot handle the compiled model
 * properly, so the execution is halted.
 *
 * The BlobReader also exposes an API required to meet the needs of all custom section readers (implemented in
 * the class inheriting "ISection").
 */
class BlobReader {
public:
    /**
     * @brief Constructs a BlobReader, associating it with the given compiled model source.
     */
    BlobReader(const ov::Tensor& source);

    /**
     * @brief Parses the given compiled model using all section readers registered so far.
     *
     * @param plugin_capabilities Indicates all capabilities of the NPU plugin. This mapping is used to evaluate the
     * CRE.
     */
    void read(const std::unordered_map<CRE::Token, std::shared_ptr<ICapability>>& plugin_capabilities);

    /**
     * @brief Register a new section reader for the given section type.
     */
    void register_reader(const SectionType type,
                         std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)> reader);

    /**
     * @brief Retrieve a parsed section.
     * @note This should be called only after "read" was invoked.
     */
    std::shared_ptr<ISection> retrieve_section(const SectionID& id);

    /**
     * @brief Retrieve the first parsed section of the given type.
     * @note This should be called only after "read" was invoked.
     * @note This function exists only for convenience. Most section types will typically have a single instance inside
     * a compiled model.
     */
    std::shared_ptr<ISection> retrieve_first_section(const SectionType section_type);

    /**
     * @brief Retrieves all parsed sections of the given type.
     * @note This should be called only after "read" was invoked.
     */
    std::optional<std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>> retrieve_sections_same_type(
        const SectionType type);

    /**
     * @brief Reads data from the compiled model source and copies it to the given destination. Also the read cursor is
     * advanced according to the given size.
     * @note This is part of the section reader API. TODO: any way to reinforce this constraint?
     */
    void copy_data_from_source(char* destination, const size_t size);

    /**
     * @brief Returns a pointer to the current position of the cursor, then advances the cursor according to the given
     * size. This method avoids copying the content of the compiled model.
     * @note This is part of the section reader API.
     */
    const void* interpret_data_from_source(const size_t size);

    /**
     * @brief Returns an RoI tensor pointing to the current position of the cursor, then advances the cursor according
     * to the given size. This method avoids copying the content of the compiled model.
     * @note This is part of the section reader API.
     */
    ov::Tensor get_roi_tensor(const size_t size);

    /**
     * @note This is part of the section reader API.
     * @returns The curent position of the read cursor, relative to the beginning of the NPU blob region.
     */
    size_t get_cursor_relative_position();

    /**
     * @note This is part of the section reader API.
     * @returns The curent position of the read cursor, relative to the beginning of the NPU blob region.
     */
    void move_cursor_to_relative_position(const size_t offset);

    /**
     * @brief Extracts the size of the NPU blob region from the given stream.
     * @details This number is a field found at the beginning of the NPU blob region.
     */
    static size_t get_npu_region_size(std::istream& stream);

    /**
     * @brief Extracts the size of the NPU blob region from the given tensor.
     * @details This number is a field found at the beginning of the NPU blob region.
     */
    static size_t get_npu_region_size(const ov::Tensor& tensor);

private:
    friend class BlobWriter;

    /**
     * @brief Where the compiled model resides.
     */
    std::reference_wrapper<const ov::Tensor> m_source;
    size_t m_npu_region_size;
    OffsetsTable m_offsets_table;
    /**
     * @brief All sections obtained after parsing the compiled model.
     */
    std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>
        m_parsed_sections;
    /**
     * @brief All known section readers that can be used to parse the compiled model.
     */
    std::unordered_map<SectionType, std::function<std::shared_ptr<ISection>(BlobReader*, const size_t)>> m_readers;

    /**
     * @brief Tracks where the blob reading was left off
     */
    size_t m_cursor;
};

}  // namespace intel_npu
