// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"

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

/**
 * @brief Creates reduced shape from input by removing dimensions.
 *
 * @param input      Input shape for reduce calculation.
 * @param axes   Reduction axes.
 * @return Reduced shape.
 */
OPENVINO_API Shape reduce(const Shape& input, const AxisSet& axes);

/**
 * @brief Creates reduced shape from input removing or replacing dimension.
 *
 * The reduction type depends on `keep_dims` flags. If it's set to true then reduced dimension will be replaced by `1`,
 * otherwise removed.
 *
 * @param input      Input shape for reduce calculation.
 * @param axes       Reduction axes.
 * @param keep_dims  Flag to keep reduced dimension.
 * @return Reduced shape.
 */
OPENVINO_API Shape reduce(const Shape& input, const AxisSet& axes, const bool keep_dims);

/**
 * @brief Creates reduced vector from input by removing elements.
 *
 * @param input  Input vector for reduce calculation.
 * @param axes   Reduction axes.
 * @return Reduced vector
 */
OPENVINO_API std::vector<size_t> reduce(const std::vector<size_t>& input, const AxisSet& axes);

/**
 * @brief Creates reduced shape from input by replacing reduced dimension with `1`.
 *
 * @param input      Input shape for reduce calculation.
 * @param axes       Reduction axes.
 * @return Reduced shape.
 */
OPENVINO_API Shape reduce_keep_dims(const Shape& input, const AxisSet& axes);

/**
 * @brief Get the broadcast shape as merge second shape into first according to broadcast specification.
 *
 * @param first           First input shape.
 * @param second          Second input shape.
 * @param broadcast_spec  Broadcast specification.
 *
 * @return Result shape from inputs with applied broadcast specification.
 */
Shape get_broadcast_shape(const Shape& first, const Shape& second, const ov::op::AutoBroadcastSpec& broadcast_spec);
}  // namespace util
}  // namespace ov
