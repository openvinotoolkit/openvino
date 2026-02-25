// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::util {

/**
 * @brief Gets size of memory in bytes for N elements of given precision.
 *
 * @param type  Element precision.
 * @param n     Number of elements.
 *
 * @return Elements size in bytes.
 */
OPENVINO_API size_t get_memory_size(const element::Type& type, const size_t n);

/**
 * @brief Gets size of memory in bytes for N elements of given precision if there is no overflow.
 *
 * @param type  Element precision.
 * @param n     Number of elements.
 * @return Memory size in bytes or std::nullopt if overflow occurs.
 */
OPENVINO_API std::optional<size_t> get_memory_size_safe(const element::Type& type, const size_t n);

/**
 * @brief Gets size of memory in bytes for shape of given precision if there is no overflow.
 *
 * @param type  Element precision.
 * @param shape Shape of elements.
 * @return Memory size in bytes or std::nullopt if overflow occurs.
 */
OPENVINO_API std::optional<size_t> get_memory_size_safe(const element::Type& type, const ov::Shape& shape);

/**
 * @brief Calculates padding size in bytes to align given position to specified alignment.
 *
 * @param alignment  The desired alignment value (bytes).
 * @param pos        Given position (address) from which padding size should be calculated (bytes).
 * @return Padding size in bytes.
 */
constexpr size_t align_padding_size(size_t alignment, size_t pos) {
    if (alignment != 0) {
        const auto pad = (pos % alignment);
        return pad == 0 ? pad : alignment - pad;
    } else {
        return 0;
    }
}
}  // namespace ov::util
