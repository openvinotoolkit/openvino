// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cinttypes>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cre.hpp"
#include "intel_npu/common/blob_reader_interface.hpp"
#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/common/offsets_table.hpp"
#include "intel_npu/common/section_type_instance_evaluator.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

/**
 * @brief Class responsible for parsing the NPU specific data of a compiled model.
 * @details There should be a 1:1 mapping between "CompiledModel" & "BlobReader" instances.
 *
 * When the user requests the importation of a model, a "BlobReader" object is created. All known section readers will
 * be registered into this object. Later, these readers will be used to parse individual sections residing within the
 * compiled model. The parsed sections can then be retrieved and used by the NPU plugin.
 *
 * During the parse procedure, a compatibility requirements expression (CRE) is evaluated as one of the first steps. If
 * the evaluation yields a negative result, then the current version of the plugin cannot handle the compiled model
 * properly, so the execution is halted.
 *
 * The BlobReader also exposes an API required to meet the needs of all custom section readers (implemented in
 * the class inheriting "ISection").
 */
class BlobReader final {
public:
    /**
     * @brief Constructs a BlobReader, associating it with the given compiled model source.
     */
    BlobReader(const ov::log::Level log_level = ov::log::Level::WARNING);

    /**
     * @brief Parses the given compiled model using all section readers registered so far.
     */
    void read(const ov::Tensor& source);

    /**
     * @brief Register a new section reader for the given section type.
     */
    void register_reader(const SectionType type, std::function<std::shared_ptr<ISection>(BlobReaderInterface&)> reader);

    void register_section_type_evaluator(const std::shared_ptr<ISectionTypeEvaluator>& evaluator);

    void register_section_type_instance_evaluate_fn(const SectionType type,
                                                    std::function<bool(BlobReaderInterface&)> function);

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
     * @brief Builds classes capable of evaluating whether or not the section instances are supported.
     * @details These evaluators are meant to be used by the CRE code while evaluating an expression. Upon encountering
     * a section type instance within the expression, the corresponding evaluator class should be retrieved and used for
     * evaluation.
     */
    std::unordered_map<SectionID, SectionTypeInstanceEvaluator> build_section_type_instance_evaluators(
        const ov::Tensor& source,
        const OffsetsTable& offsets_table,
        const size_t npu_region_size) const;

    /**
     * @brief All sections obtained after parsing the compiled model.
     */
    std::unordered_map<SectionType, std::unordered_map<SectionTypeInstance, std::shared_ptr<ISection>>>
        m_parsed_sections;

    /**
     * @brief The order in which the sections have been parse.
     * @note This order is used only to ensure idempotency. A BlobWriter can be built using the sections parsed by a
     * BlobReader. By making use of this order, the BlobWriter is able to export a blob that has the exact same content
     * the original one had.
     */
    std::vector<SectionID> m_parsed_sections_order;

    /**
     * @brief All known section readers that can be used to parse the compiled model.
     */
    std::unordered_map<SectionType, std::function<std::shared_ptr<ISection>(BlobReaderInterface&)>> m_readers;
    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> m_section_type_evaluators;
    std::unordered_map<SectionType, std::function<bool(BlobReaderInterface&)>> m_section_type_instance_evaluate_fn;

    Logger m_logger;
};

}  // namespace intel_npu
