// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace util {
/// \brief Normalize value to the max if value is negative.
///
/// \param value  Input value to normalize.
/// \param max    Value used for normalization
///
/// \return Value if positive otherwise return value + max
OPENVINO_API int64_t normalize(const int64_t& value, const int64_t& max);

/// \brief Check if values in vector are unique.
///
/// \param data  Input data to check.
///
/// \return True if unique otherwise false.
OPENVINO_API bool are_unique(const std::vector<int64_t>& data);

/// \brief Clip value to minimum if below min, or to maximum of above max.
///
/// \param value  Value to be clipped.
/// \param min    Minimum value bound.
/// \param max    Maximum value bound.
///
/// \return Value if between min, max otherwise min or max.
OPENVINO_API int64_t clip(const int64_t& value, const int64_t& min, const int64_t& max);

/// \brief Constant folds a subgraph to a constant node
///
/// \param subgraph sink
///
/// \return Constant node or nullptr if unable to constant fold the subgraph
OPENVINO_API std::shared_ptr<op::v0::Constant> constantfold_subgraph(const Output<Node>& subgraph_sink);

/**
 * @brief Runs an estimation of source tensor. If it succeeded to calculate both bounds and
 * they are the same returns Constant operation from the resulting bound, otherwise nullptr.
 *
 * @param source  Node output used to get its tensor data as constant.
 * @return Shared pointer to constant data or nullptr.
 */
OPENVINO_API std::shared_ptr<op::v0::Constant> get_constant_from_source(const Output<Node>& source);

/**
 * @brief Make scalar tensor which stores maximum value of ov::element::Type.
 *
 * @param et  Element type to get its maximum.
 * @return Tensor with maximum value.
 */
Tensor make_tensor_of_max_value(const element::Type_t et);
}  // namespace util
}  // namespace ov
