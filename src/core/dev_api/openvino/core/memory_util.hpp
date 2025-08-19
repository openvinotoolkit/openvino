// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::util {

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
}  // namespace ov::util
