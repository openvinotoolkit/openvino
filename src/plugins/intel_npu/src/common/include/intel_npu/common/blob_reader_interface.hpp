// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "blob_source.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

class BlobReaderInterface final {
public:
    /**
     * @brief Constructs a BlobReader, associating it with the given compiled model source.
     */
    BlobReaderInterface(BlobSource& source,
                        const size_t npu_region_start,
                        const size_t npu_region_size,
                        const size_t section_start,
                        const size_t section_length,
                        const FilteredConfig& config);

    /**
     * @brief Reads data from the compiled model source and copies it to the given destination. Also the read cursor is
     * advanced according to the given size.
     */
    void read_into_buffer(void* destination, const size_t size);

    /**
     * @brief Returns a pointer to the current position of the cursor, then advances the cursor according to the given
     * size. This method avoids copying the content of the compiled model.
     */
    const void* read_view(const size_t size);
    // TODO implement is_contiguous

    /**
     * @brief Returns an RoI tensor pointing to the current position of the cursor, then advances the cursor according
     * to the given size. This method avoids copying the content of the compiled model.
     */
    ov::Tensor create_roi_tensor(const size_t size);

    size_t get_offset_relative_to_current_section() const;

    void move_cursor_relative_to_current_section(const size_t offset);

    size_t get_offset_relative_to_npu_region() const;

    void move_cursor_relative_to_npu_region(const size_t offset);

    bool source_is_contiguous() const;

    size_t get_section_length() const;

    FilteredConfig get_config() const;

    ov::log::Level get_log_level() const;

private:
    std::reference_wrapper<BlobSource> m_source;

    size_t m_npu_region_start;
    size_t m_section_start;
    size_t m_section_end;

    FilteredConfig m_config;
    Logger m_logger;
};

}  // namespace intel_npu
