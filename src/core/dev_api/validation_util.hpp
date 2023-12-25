// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/attr_types.hpp"

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

/// \brief Runs an estimation of source tensor. If it succeeded to calculate both bounds and
/// they are the same returns Constant operation from the resulting bound, otherwise nullptr.
///
/// \param source  Node output used to get its tensor data as constant.
/// \return Shared pointer to constant data or nullptr.
OPENVINO_API std::shared_ptr<op::v0::Constant> get_constant_from_source(const Output<Node>& source);

/// \brief Make scalar tensor which stores maximum value of ov::element::Type.
/// \param et  Element type to get its maximum.
/// \return Tensor with maximum value.
Tensor make_tensor_of_max_value(const element::Type_t et);

/// \brief Make scalar tensor which stores minimum value of ov::element::Type.
/// \param et  Element type to get its minimum.
/// \return Tensor with minimum value.
Tensor make_tensor_of_min_value(const element::Type_t et);

/// \brief Apply auto padding to padding_above and padding_below inputs
///        if all needed informations are known.
///
/// \param image_shape       The shape of input image.
/// \param filter_shape      The shape of filter input.
/// \param filter_strides    The strides of applied padding.
/// \param filter_dilations  The dilations of applied padding.
/// \param pad_type          The type of padding. Auto padding is applied only
///                          for SAME_UPPER and SAME_LOWER mode.
/// \param padding_above     The beginning of padding shape.
/// \param end               The beginning of padding shape.
///
/// \return true if auto padding was applied successfully (all needed informations such as
///         spatial dims are known), false otherwise.
OPENVINO_DEPRECATED("This function is deprecated and will be removed.")
OPENVINO_API bool try_apply_auto_padding(const PartialShape& image_shape,
                                         const Shape& filter_shape,
                                         const Strides& filter_strides,
                                         const Strides& filter_dilations,
                                         const op::PadType pad_type,
                                         CoordinateDiff& padding_above,
                                         CoordinateDiff& padding_below);

/// @brief Get the tensors shapes as ov::PartialShape.
///
/// @param tensors  Input tensors vector to get their shapes.
/// @return Vector of partial shapes same size as input tensor vector.
OPENVINO_API std::vector<PartialShape> get_tensors_partial_shapes(const TensorVector& tensors);

/// \brief Get the node input partial shapes.
///
/// \param node   Node to extract input shapes.
///
/// \return Vector of PartialShapes of each input.
OPENVINO_API std::vector<PartialShape> get_node_input_partial_shapes(const ov::Node& node);

/// \brief Check if rank is compatible to any of others ranks.
///
/// \param r       Rank to check.
/// \param others  List of ranks used to check compatibility.
///
/// \return True if rank compatible to any of others, otherwise false.
OPENVINO_API bool is_rank_compatible_any_of(const Rank& r, std::initializer_list<Rank> others);

/// \brief Infers the output batch shape for convolution forward propagation.
///
/// \return Infered output shape.
OPENVINO_DEPRECATED("This function is deprecated and will be removed.")
OPENVINO_API PartialShape infer_convolution_forward(const Node* node,
                                                    const PartialShape& data_batch_shape,
                                                    const Strides& data_dilation,
                                                    const CoordinateDiff& data_padding_below,
                                                    const CoordinateDiff& data_padding_above,
                                                    const PartialShape& filters_shape,
                                                    const Strides& filter_strides,
                                                    const Strides& filter_dilation);

/// \brief Infers image paddings.
OPENVINO_DEPRECATED("This function is deprecated and will be removed.")
OPENVINO_API void infer_auto_padding(const Shape& image_shape,
                                     const Shape& filter_shape,
                                     const Strides& filter_strides,
                                     const Strides& filter_dilations,
                                     const op::PadType pad_type,
                                     CoordinateDiff& padding_above,
                                     CoordinateDiff& padding_below);

/// \brief Evaluates lower and upper value estimations for the output tensor. Estimation would be represented as partial
/// shape object using Dimension(min, max) for each element.
///
/// \param output Node output pointing to the tensor for estimation.
/// \param pshape Resulting estimation would be stored in this PartialShape.
///
/// \return True if estimations evaluation was successful, false otherwise.
OPENVINO_API bool evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape);

/// \brief Propagates value label from 0 input to the only output through an operation. Not applicable for operations
/// which require values interaction (example: mathematical operations). Could be used for movement operations (example:
/// gathering, shape change)
///
/// \param node Operation to be performed
/// \param output_labels Vector of TensorLabel objects representing resulting value labels
///
/// \return True if label evaluation was successful, false otherwise.
OPENVINO_API bool default_label_evaluator(const Node* node, TensorLabelVector& output_labels);

/// \brief Generates default order of axes transposition at the end of input vector.
///
/// The default axes order is a descending sequence of numbers starting at `length - 1`.
///
/// \param axes_order  Vector where default order will be generated.
/// \param length      Sequence length of axes order.
OPENVINO_API void generate_transpose_default_order(std::vector<int64_t>& axes_order, size_t length);

/// \brief Checks whether axes order has valid values.
///
/// Axes order has to be a set of unique numbers in range [0, size).
///
/// \param axes_order  Vector with axes order to check.
/// \param size        Input for transpose rank size.
///
/// \return True if axes order is valid, false otherwise.
OPENVINO_API bool is_valid_axes_order(const std::vector<int64_t>& axes_order, size_t size);

/// \brief Checks whether label tensor has no labels.
///
/// \param labels  Label tensor to check.
/// \return True if there are no labels, false otherwise.
OPENVINO_API bool has_no_labels(const TensorLabel& labels);

/// \brief      Handles out of range axis.
///
/// \param[in]  node         The node with requested axis.
/// \param[in]  axis         The requested axis value.
/// \param[in]  tensor_rank  The corresponding tensor rank.
///
/// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
///            returns error. If negative axis, it counts from the last to the first axis,
///            by adding tensor_rank to axis.
OPENVINO_API int64_t normalize_axis(const Node* node, std::int64_t axis, const Rank& tensor_rank);

/// \brief      Handles out of range axis.
///
/// \param[in]  node_description   The node with requested axis.
/// \param[in]  axis               The requested axis value.
/// \param[in]  tensor_rank        The corresponding tensor rank.
///
/// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
///            returns error. If negative axis, it counts from the last to the first axis,
///            by adding tensor_rank to axis.
OPENVINO_API int64_t normalize_axis(const std::string& node_description, std::int64_t axis, const Rank& tensor_rank);

/// \brief      Handles out of range axis.
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
OPENVINO_API int64_t normalize_axis(const Node* node,
                                    std::int64_t axis,
                                    std::uint64_t tensor_rank,
                                    std::int64_t axis_range_min,
                                    std::int64_t axis_range_max);

/// \brief      Handles out of range axis.
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
OPENVINO_API int64_t normalize_axis(const std::string& node_description,
                                    std::int64_t axis,
                                    std::uint64_t tensor_rank,
                                    std::int64_t axis_range_min,
                                    std::int64_t axis_range_max);

/// \brief      Handles out of range axes in vector.
///
/// \param[in]  node_description  The name of node with requested axes.
/// \param[in]  axes              The requested vector of axes.
/// \param[in]  tensor_rank       The corresponding tensor rank.
///
/// \return     If any negative axis in vector, it counts from the last to the first
///             axis, by adding tensor_rank to axis.
OPENVINO_API std::vector<size_t> normalize_axes(const std::string& node_description,
                                                const std::vector<int64_t>& axes,
                                                const Rank& tensor_rank);

/// \brief      Handles out of range axes in vector.
/// If any negative axis in vector, it counts from the last to the first axis,
/// by adding tensor_rank to axis. Changes axes vector inplace.
///
/// \param[in]      node         The node with requested axes.
/// \param[in]      tensor_rank  The corresponding tensor rank.
/// \param[in,out]  axes         The requested vector of axes.
OPENVINO_API void normalize_axes(const Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes);
}  // namespace util
}  // namespace ov
