// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov {

OPENVINO_API
PartialShape infer_convolution_forward(const Node* node,
                                       const PartialShape& data_batch_shape,
                                       const Strides& data_dilation,
                                       const CoordinateDiff& data_padding_below,
                                       const CoordinateDiff& data_padding_above,
                                       const PartialShape& filters_shape,
                                       const Strides& filter_strides,
                                       const Strides& filter_dilation);

OPENVINO_API
void infer_auto_padding(const Shape& image_shape,
                        const Shape& filter_shape,
                        const Strides& filter_strides,
                        const Strides& filter_dilations,
                        const op::PadType pad_type,
                        CoordinateDiff& padding_above,
                        CoordinateDiff& padding_below);

/// \brief Normalize value to the max if value is negative.
///
/// \param value  Input value to normalize.
/// \param max    Value used for normalization
///
/// \return Value if positive otherwise return value + max
OPENVINO_API
int64_t normalize(const int64_t& value, const int64_t& max);

/// \brief      Handle out of range axis.
///
/// \param[in]  node         The node with requested axis.
/// \param[in]  axis         The requested axis value.
/// \param[in]  tensor_rank  The corresponding tensor rank.
///
/// \return    Checking if axis is in range [-tensor_rank, tensor_rank-1], otherwise
///            returns error. If negative axis, it counts from the last to the first axis,
///            by adding tensor_rank to axis.
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
OPENVINO_API
void normalize_axes(const Node* node, const int64_t& tensor_rank, std::vector<int64_t>& axes);

/// \brief Evaluates lower and upper value estimations for the output tensor. Estimation would
/// be represented as partial shape object using Dimension(min, max) for each element.
/// \param output Node output pointing to the tensor for estimation.
/// \param pshape Resulting estimation would be stored in this PartialShape.
/// \return boolean status if value evaluation was successful.
OPENVINO_API bool evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape);

/// \brief Runs an estimation of source tensor. If it succeeded to calculate both bounds and
/// they are the same returns Constant operation from the resulting bound, otherwise nullptr.
OPENVINO_API std::shared_ptr<op::v0::Constant> get_constant_from_source(const Output<Node>& source);

/// \brief Propagates value label from 0 input to the only output through an operation.
/// Not applicable for operations which require values interaction (example: mathematical
/// operations). Could be used for movement operations (example: gathering, shape change)
/// \param node Operation to be performed
/// \param output_labels Vector of TensorLabel objects representing resulting value labels
/// \return boolean status if label evaluation was successful.
OPENVINO_API bool default_label_evaluator(const Node* node, TensorLabelVector& output_labels);

/// \brief Generates transpose default axes order at end of input vector.
///
/// Default axes order is decreasing sequence numbers which start from `length - 1`.
///
/// \param axes_order  Vector where default order will be generated.
/// \param length      Sequence length of axes order.
OPENVINO_API void generate_transpose_default_order(std::vector<int64_t>& axes_order, const size_t length);

/// \brief Check if vector of axes order has got valid values.
///
/// Axes order has to be unique numbers in range of [0, size).
///
/// \param axes_order  Vector with axes order to check.
/// \param size        Input for transpose rank size.
///
/// \return true if axes order is valid otherwise false.
OPENVINO_API bool is_valid_axes_order(const std::vector<int64_t>& axes_order, const size_t size);

/// \brief Checks label tensor if there is no label
///
/// \param labels  Label tensor for check.
/// \return True if there is no labels, otherwise false.
OPENVINO_API bool has_no_labels(const TensorLabel& labels);

/// \brief Get the node input partial shapes.
///
/// \param node   Node to extract input shapes.
///
/// \return Vector of PartialShapes of each input.
OPENVINO_API std::vector<PartialShape> get_node_input_partial_shapes(const ov::Node& node);

/// \brief Check if rank is compatible to any of rank from container.
///
/// \param rank   Rank to check.
/// \param ranks  VEctor of ranks used to check input rank compatibility.
///
/// \return True if rank compatible to any from ranks, otherwise false.
OPENVINO_API bool is_rank_compatible_any_of(const ov::Rank& rank, const std::vector<ov::Rank>& ranks);

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
/// \param max    Maximum value boiund
///
/// \return Value if between min, max otherwise min or max.
OPENVINO_API int64_t clip(const int64_t& value, const int64_t& min, const int64_t& max);

OPENVINO_API bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

/// \brief Checks if all the elements of the bound Tensor are positive
OPENVINO_API bool tensor_is_positive(const Tensor& bound);

/// \brief Estimates upper bound for node output tensors using only upper bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of Tensors representing resulting upper value estimations
/// \return boolean status if value evaluation was successful.
OPENVINO_API bool default_upper_bound_evaluator(const Node* node, TensorVector& output_values);

/// \brief Estimates lower bound for node output tensors using only lower bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of Tensors representing resulting lower value estimations
/// \return boolean status if value evaluation was successful.
OPENVINO_API bool default_lower_bound_evaluator(const Node* node, TensorVector& output_values);

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param Node output pointing to the tensor for estimation.
/// \return Tensor to estimated value.
OPENVINO_API Tensor evaluate_lower_bound(const Output<Node>& output);

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param output Tensor to be estimated.
/// \return Tensor to estimated value.
OPENVINO_API Tensor evaluate_upper_bound(const Output<Node>& output);

/// \brief Evaluates lower and upper value estimations of the output tensor. Traverses graph up
/// to deduce estimation through it.
/// \param output Node output pointing to the tensor for estimation.
/// \return pair with Tensors for lower and upper value estimation.
OPENVINO_API std::pair<Tensor, Tensor> evaluate_both_bounds(const Output<Node>& output);

/// \brief Estimates both bounds for node output tensors using both bounds of inputs. Works for
/// operations with two inputs (in_1 and in_2). Brute forces all the pairs of bounds for inputs
/// and evaluates all of them: {in_1_lower, in_2 lower}, {in_1_lower, in_2 upper}, {in_1_upper,
/// in_2_lower}, {in_1_upper, in_2_upper}. Lower and upper values are selected from all the
/// outputs calculated using input pairs.
///
/// \param node Operation to be performed
/// \param lower_output_values Vector of Tensors representing resulting lower value estimations
/// \param upper_output_values Vector of Tensors representing resulting upper value estimations
/// \return boolean status if value evaluation was successful.
OPENVINO_API bool interval_bound_evaluator(const Node* node,
                                           TensorVector& lower_output_values,
                                           TensorVector& upper_output_values);

/// \brief Constant folds a subgraph to a constant node
///
/// \param subgraph sink
///
/// \return Constant node or nullptr if unable to constantfold the subgraph
OPENVINO_API std::shared_ptr<op::v0::Constant> constantfold_subgraph(const Output<Node>& subgraph_sink);
}  // namespace ov
