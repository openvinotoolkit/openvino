// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {
namespace util {

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

/**
 * @brief Normalize shape index to the rank
 *
 * If input index is out of range [-rank, rank) throws exception.
 *
 * @param idx   Shape dimension index.
 * @param rank  Shape rank.
 * @return Normalized shape dimension index.
 */
OPENVINO_API std::ptrdiff_t normalize_shape_index(std::ptrdiff_t idx, size_t rank);
}  // namespace util
}  // namespace ov
