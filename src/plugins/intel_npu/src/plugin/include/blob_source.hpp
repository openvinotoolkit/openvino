// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <variant>

#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/runtime/tensor.hpp"

namespace intel_npu {

/**
 * @brief Class that allows reading from either an "std::istream" or "ov::Tensor" object using a single set of common
 * methods.
 * @details The "stream" type is handled using the "std" API. The tensor is handled by managing a data cursor. Only the
 * tensor type supports extracting data without copying, since the "stream" type is not guaranteed to have a contiguous
 * buffer under the hood.
 */
class BlobSource {
public:
    explicit BlobSource(std::istream& source, const ov::log::Level log_level = Logger::global().level());

    explicit BlobSource(const ov::Tensor& source, const ov::log::Level log_level = Logger::global().level());

    /**
     * @brief Copies data from the blob source to the given destination.
     * @details The data is copied from where the data cursor currently points to; the cursor is then advanced by "size"
     * bytes.
     *
     * @param destination The data will be copied into this buffer
     * @param size The amount of data that will be copied. The data cursor will also be advanced by this amount.
     */
    void read_into_buffer(void* destination, const size_t size);

    /**
     * @brief Extracts data from the blob source without copying it.
     * @details The returned address is the position where the data cursor currently points to. Before returning, the
     * cursor is also advanced by "size" bytes.
     * @throws ov::Exception if the underlying type does not have a guaranteed contiguous buffer (e.g. "std::istream").
     * @note "BlobSource::is_contiguous" can be used to determine if this function can be called safely.
     *
     * @param size The data cursor will be advanced by this amount.
     * @return The blob content location where the data cursor currently points to (prior to advancing the cursor).
     */
    const void* read_view(const size_t size);

    /**
     * @brief Extracts a region-of-interest tensor from the blob source without copying it.
     * @details The tensor is a view towards the content of the blob, delimited by [data cursor:data cursor + size).
     * Before returning, the cursor is also advanced by "size" bytes.
     * @throws ov::Exception if the underlying type does not have a guaranteed contiguous buffer (e.g. "std::istream").
     * @note "BlobSource::is_contiguous" can be used to determine if this function can be called safely.
     *
     * @param size The data cursor will be advanced by this amount.
     * @return A region-of-interest tensor as a view towards the content of the blob, delimited by [data cursor:data
     * cursor + size).
     */
    ov::Tensor create_roi_tensor(const size_t size);

    /**
     * @brief Moves the data cursor by "offset" bytes starting from the given reference.
     *
     * @param offset The amount of bytes the cursor will be moved.
     * @param reference "std::ios::beg", "std::ios::cur" and "std::ios::end" are the only supported values. These
     * correspond to the beginning of the blob source, the current position of the cursor and the end of the source.
     */
    void seekg(const int64_t offset, const std::ios_base::seekdir reference = std::ios::beg);

    /**
     * @return The current position of the data cursor
     */
    size_t tellg() const;

    /**
     * @return The total size of the blob
     */
    size_t get_total_size() const;

    /**
     * @return The remaining size of the blob, starting from the current position of the data cursor
     */
    size_t get_remaining_size() const;

    /**
     * @return true The blob is contiguous (the "tensor" scenario).
     * @return false Otherwise (the "stream" scenario)
     */
    bool is_contiguous() const;

private:
    /**
     * @brief A union that captures all possible data types of the blob. The tensor type also has a separate data cursor
     * attached to it.
     */
    std::variant<std::reference_wrapper<std::istream>, std::pair<std::reference_wrapper<const ov::Tensor>, size_t>>
        m_source;
    size_t m_size;

    Logger m_logger;
};

}  // namespace intel_npu
