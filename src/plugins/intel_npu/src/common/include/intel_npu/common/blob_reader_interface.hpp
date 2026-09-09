// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "intel_npu/common/blob_source.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/isection_type_evaluator.hpp"
#include "intel_npu/utils/logger/logger.hpp"

namespace intel_npu {

class BlobReaderInterface final {
public:
    /**
     * @brief Constructs an interface that allows the section readers to read data from the blob in a restricted manner.
     * @details By using an instantiation of this class, reading can only happen within the given boundaries.
     *
     * @param source The data will be read from here
     * @param npu_region_start The offset within the source where the NPU specific data begins
     * @param npu_region_size The size of the NPU specific region within the source
     * @param section_start The offset within the source where the section begins
     * @param section_length The size of the current section within the source
     * @param config An optional configuration that may be queried by the section readers leveraging this interface
     */
    BlobReaderInterface(BlobSource& source,
                        const size_t npu_region_start,
                        const size_t npu_region_size,
                        const size_t section_start,
                        const size_t section_length,
                        const std::optional<FilteredConfig>& config = std::nullopt);

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

    std::optional<FilteredConfig> get_config() const;

    ov::log::Level get_log_level() const;

private:
    std::reference_wrapper<BlobSource> m_source;

    size_t m_npu_region_start;
    size_t m_section_start;
    size_t m_section_end;

    std::optional<FilteredConfig> m_config;
    Logger m_logger;
};

}  // namespace intel_npu
