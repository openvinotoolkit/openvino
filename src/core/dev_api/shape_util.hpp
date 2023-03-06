// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace util {
/**
 * @brief Makes spacial version of 2D ov::Shape which is recognize as dynamic.
 *
 * This is special case used for tensor <-> host tensor conversion to indicate that tensor got dynamic shape.
 *
 * @return 2-D shape with {0, SIZE_MAX}
 */
OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
OPENVINO_API Shape make_dynamic_shape();

/**
 * @brief Check if Shape is marked as dynamic.
 *
 * @param s  Shape for check.
 * @return True if shape is dynamic otherwise false.
 */
OPENVINO_DEPRECATED("This function is deprecated and will be removed soon.")
OPENVINO_API bool is_dynamic_shape(const Shape& s);
}  // namespace util
}  // namespace ov
