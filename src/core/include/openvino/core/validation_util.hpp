// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {

/// \brief      Handle out of range axis.
///
/// \param[in]  node         The node with requested axis.
/// \param[in]  axis         The requested axis value.
/// \param[in]  tensor_rank  The corresponding tensor rank.
///
/// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
///            returns error. If negative axis, it counts from the last to the first axis,
///            by adding tensor_rank to axis.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
int64_t normalize_axis(const Node* node, std::int64_t axis, const Rank& tensor_rank);

/// \brief      Handle out of range axes in vector.
///
/// \param[in]  node_description  The name of node with requested axes.
/// \param[in]  axes              The requested vector of axes.
/// \param[in]  tensor_rank       The corresponding tensor rank.
///
/// \return     If any negative axis in vector, it counts from the last to the first
///             axis, by adding tensor_rank to axis.
///
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
std::vector<size_t> normalize_axes(const std::string& node_description,
                                   const std::vector<int64_t>& axes,
                                   const Rank& tensor_rank);

/// \brief      Handle out of range axis.
///
/// \param[in]  node_description   The node with requested axis.
/// \param[in]  axis               The requested axis value.
/// \param[in]  tensor_rank        The corresponding tensor rank.
///
/// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
///            returns error. If negative axis, it counts from the last to the first axis,
///            by adding tensor_rank to axis.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
int64_t normalize_axis(const std::string& node_description, std::int64_t axis, const Rank& tensor_rank);

/// \brief      Handle out of range axis.
///
/// \param[in]  node            The node with requested axis.
/// \param[in]  axis            The requested axis value.
/// \param[in]  tensor_rank     The corresponding tensor rank.
/// \param[in]  axis_range_min  The min value of accepted range for axis.
/// \param[in]  axis_range_max  The max value of accepted range for axis.
///
/// \return     Checking if axis is in range [axis_range_min, axis_range_max], otherwise
///             returns error. If negative axis, it counts from the last to the first axis,
///             by adding tensor_rank to axis.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
int64_t normalize_axis(const Node* node,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max);

/// \brief      Handle out of range axis.
///
/// \param[in]  node_description   The name of node with requested axis.
/// \param[in]  axis               The requested axis value.
/// \param[in]  tensor_rank        The corresponding tensor rank.
/// \param[in]  axis_range_min     The min value of accepted range for axis.
/// \param[in]  axis_range_max     The max value of accepted range for axis.
///
/// \return     Checking if axis is in range [axis_range_min, axis_range_max], otherwise
///             returns error. If negative axis, it counts from the last to the first axis,
///             by adding tensor_rank to axis.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
int64_t normalize_axis(const std::string& node_description,
                       std::int64_t axis,
                       std::uint64_t tensor_rank,
                       std::int64_t axis_range_min,
                       std::int64_t axis_range_max);

/// \brief      Handle out of range axes in vector.
/// If any negative axis in vector, it counts from the last to the first axis,
/// by adding tensor_rank to axis. Changes axes vector inplace.
///
/// \param[in]      node         The node with requested axes.
/// \param[in]      tensor_rank  The corresponding tensor rank.
/// \param[in,out]  axes         The requested vector of axes.
///
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API
void normalize_axes(const Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes);

/// \brief Evaluates lower and upper value estimations for the output tensor. Estimation would
/// be represented as partial shape object using Dimension(min, max) for each element.
/// \param output Node output pointing to the tensor for estimation.
/// \param pshape Resulting estimation would be stored in this PartialShape.
/// \return boolean status if value evaluation was successful.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API bool evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape);

/// \brief Runs an estimation of source tensor. If it succeeded to calculate both bounds and
/// they are the same returns Constant operation from the resulting bound, otherwise nullptr.
OPENVINO_DEPRECATED("This function is deprecated and will be moved to dev api in 2024.0 release.")
OPENVINO_API std::shared_ptr<op::v0::Constant> get_constant_from_source(const Output<Node>& source);
}  // namespace ov
