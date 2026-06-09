// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

class BlobReaderInterface final {
public:
    /**
     * @brief Constructs a BlobReader, associating it with the given compiled model source.
     */
    BlobReaderInterface(
        const ov::Tensor& source,
        const size_t section_start,
        const size_t section_length,
        const size_t npu_region_size,
        const std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>>& section_type_evaluators,
        const ov::log::Level log_level = ov::log::Level::WARNING);

    /**
     * @brief Reads data from the compiled model source and copies it to the given destination. Also the read cursor is
     * advanced according to the given size.
     */
    void copy_data_from_source(char* destination, const size_t size);

    /**
     * @brief Returns a pointer to the current position of the cursor, then advances the cursor according to the given
     * size. This method avoids copying the content of the compiled model.
     */
    const void* interpret_data_from_source(const size_t size);

    /**
     * @brief Returns an RoI tensor pointing to the current position of the cursor, then advances the cursor according
     * to the given size. This method avoids copying the content of the compiled model.
     */
    ov::Tensor get_roi_tensor(const size_t size);

    size_t get_offset_relative_to_current_section() const;

    void move_cursor_relative_to_current_section(const size_t offset);

    size_t get_offset_relative_to_npu_region() const;

    void move_cursor_relative_to_npu_region(const size_t offset);

    size_t get_section_length() const;

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> get_section_type_evaluators() const;

    ov::log::Level get_log_level() const;

private:
    std::reference_wrapper<const ov::Tensor> m_source;

    /**
     * @brief Tracks where the blob reading was left off
     */
    size_t m_cursor;

    size_t m_section_start;
    size_t m_section_end;

    std::unordered_map<SectionType, std::shared_ptr<ISectionTypeEvaluator>> m_section_type_evaluators;

    Logger m_logger;
};

}  // namespace intel_npu
