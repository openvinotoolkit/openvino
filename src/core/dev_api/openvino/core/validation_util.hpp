// Copyright (C) 2018-2025 Intel Corporation
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
OPENVINO_API int64_t normalize(const int64_t value, const int64_t max);

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
OPENVINO_API Tensor make_tensor_of_max_value(const element::Type_t et);

/// \brief Make scalar tensor which stores minimum value of ov::element::Type.
/// \param et  Element type to get its minimum.
/// \return Tensor with minimum value.
OPENVINO_API Tensor make_tensor_of_min_value(const element::Type_t et);

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

/// \brief Evaluates lower and upper value estimations for the output tensor. Estimation would be represented as partial
/// shape object using Dimension(min, max) for each element.
///
/// \param output Node output pointing to the tensor for estimation.
/// \param pshape Resulting estimation would be stored in this PartialShape.
///
/// \return True if estimations evaluation was successful, false otherwise.
OPENVINO_API bool evaluate_as_partial_shape(const Output<Node>& output, PartialShape& pshape);

/// \brief Propagates value sumbol from 0 input to the only output through an operation. Not applicable for operations
/// which require values interaction (example: mathematical operations). Could be used for movement operations (example:
/// gathering, shape change)
///
/// \param node Operation to be performed
/// \param output_symbols Vector of TensorSymbol objects representing resulting value symbols
///
/// \return True if symbol evaluation was successful, false otherwise.
OPENVINO_API bool default_symbol_evaluator(const Node* node, TensorSymbolVector& output_symbols);

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

/// \brief Checks whether symbol tensor has no symbols.
///
/// \param symbols  Symbol tensor to check.
/// \return True if there are no symbols, false otherwise.
OPENVINO_API bool has_no_symbols(const TensorSymbol& symbols);

/// \brief Checks if axis value is in rank range [-rank, rank).
/// \note For scalar rank axis can be only 0.
///
/// \param axis  Axis value.
/// \param rank  Rank value.
/// \return      True if axis within rank range.
OPENVINO_API bool is_axis_valid(const int64_t axis, const int64_t rank);

/// \brief Validate axis if is in withing rank
///
/// Throws if rank is dynamic or, axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param axis  Axis value to be checked.
/// \param rank  Rank used for axis validation.
/// \param node  Node use for detailed error message.
OPENVINO_API void validate_axis(const int64_t axis, const Rank& rank, const Node& node);

/// \brief Normalize axis against the rank.
/// \note  No input validation.
///
/// \param axis  Axis value to be normalized.
/// \param rank  Rank value used for axis normalization.
/// \return      Normalized axis value.
OPENVINO_API size_t normalize_axis(const int64_t axis, const int64_t rank);

/// \brief Tries normalize axis against the rank.
///
/// Throws if rank is dynamic or, axis outside rank range [-rank, rank).
///
/// \param axis  Axis value to be normalized.
/// \param rank  Rank used for axis normalization.
/// \return      Normalized axis value.
OPENVINO_API size_t try_normalize_axis(const int64_t axis, const Rank& rank);

/// \brief Normalize axis against the rank.
///
/// Throws if rank is dynamic or, axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param axis  Axis value to be normalized.
/// \param rank  Rank used for axis normalization.
/// \param node  Node use for detailed error message.
/// \return      Normalized axis value.
OPENVINO_API size_t try_normalize_axis(const int64_t axis, const Rank& rank, const Node& node);

/// \brief Validate axes if are in withing rank
///
/// Throws if rank is dynamic or any axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param axes  Axes value to be checked.
/// \param rank  Rank used for axes validation.
/// \param node  Node use for detailed error message.
OPENVINO_API void validate_axes(const std::vector<int64_t>& axes, const Rank& rank, const Node& node);

/// \brief Normalize axes vector against the rank.
/// \note  No input validation.
///
/// \param axes  Axes which will be normalized (in-place).
/// \param rank  Rank value used for axes normalization.
OPENVINO_API void normalize_axes(std::vector<int64_t>& axes, const int64_t rank);

/// \brief Normalize axes against the rank.
///
/// Throws if rank is dynamic or any axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param axes  Axes which will be normalized (in-place).
/// \param rank  Rank used for axes normalization.
/// \param node  Node use for detailed error message.
/// \return
OPENVINO_API void try_normalize_axes(std::vector<int64_t>& axes, const Rank& rank, const Node& node);

/// \brief Get the normalized axes as ov::AxisVector from tensor data.
///
/// Throws if rank is dynamic or any axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param tensor  Tensor with axes for normalization.
/// \param rank    Rank value to normalize axes.
/// \param node    Node use for detailed error message.
/// \return        Normalized AxisVector.
OPENVINO_API AxisVector try_get_normalized_axis_vector(const Tensor& tensor, const Rank& rank, const Node& node);

/// \brief Get the normalized axes as ov::AxisSet from raw tensor data.
///
/// Throws if rank is dynamic or any axis outside rank range [-rank, rank). The error message has detailed
/// information about node.
///
/// \param tensor  Tensor with axes for normalization.
/// \param rank    Rank value to normalize axes.
/// \param node    Node use for detailed error message.
/// \return        Normalized AxisSet.
OPENVINO_API AxisSet try_get_normalized_axis_set(const Tensor& tensor, const Rank& rank, const Node& node);

}  // namespace util
}  // namespace ov
