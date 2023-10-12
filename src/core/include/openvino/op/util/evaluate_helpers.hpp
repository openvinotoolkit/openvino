// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/partial_shape.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace util {

/**
 * @brief Get the tensors shapes as ov::PartialShape.
 *
 * @param tensors  Input tensors vector to get its shapes.
 * @return Vector of partial shapes sam size as input tensor vector.
 */
std::vector<PartialShape> get_tensors_partial_shapes(const TensorVector& tensors);
}  // namespace util
}  // namespace op
}  // namespace ov
