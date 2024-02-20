// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/core/rank.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace op {
namespace util {

/**
 * @brief Get the normalized axes as ov::AxisSet from raw tensor data.
 *
 * @param node    A node pointer used for detailed description if normalization fails.
 * @param tensor  Tensor with axes for normalization.
 * @param rank    Rank value to normalize axes.
 * @return        Normalized axes as set.
 */
AxisSet get_normalized_axes_from_tensor(const Node* const node, const Tensor& tensor, const Rank& rank);
}  // namespace util
}  // namespace op
}  // namespace ov
