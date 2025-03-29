// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace reshape {
/// \brief      Infer `output_shape` dimension values.
///
/// \par Inference rules
///     \li         The input_shape may consist at most on -1 value. In this case the
///                 value is inferred from the size of the tensor and the remaining
///                 dimensions.
///     \li         If a dimension value is equal to 0, then its output value is going
///                 to be copied from the input_shape argument.
///
/// \param[in]  node_name     The node name.
/// \param[in]  input_shape   The input node shape.
/// \param[in]  output_shape  The requested output shape for the input node data.
///
/// \return     A vector containing new, valid node shape.
///
std::vector<std::size_t> infer_dimensions(const std::string& node_name,
                                          const std::vector<std::size_t>& input_shape,
                                          const std::vector<std::size_t>& output_shape);

/// \brief      Handle a node which represents a scalar value.
///
/// \note       Some ONNX nodes, which should provide scalar values are given as
///             tensors of shape {1}. This function will provide a reshape of
///             such a node with Shape{1} into a scalar with Shape{}.
///
/// \param[in]  node   Node to reshape.
///
/// \return     Original node or a node representing a reshape of the original.
///
ov::Output<ov::Node> interpret_as_scalar(const ov::Output<ov::Node>& node);

/// \brief      Reshape node from shape {C} to {1, C, 1, 1,...}
///
/// \note       This function will reshape the input node
///             with a shape of {C} into a node with Shape{1, C, 1, 1, ..}.
///             The most common input to this function would be scale or bias to
///             BatchNorm or bias to Conv.
///
/// \param[in]  node            Node to reshape.
/// \param[in]  expected_rank   Expected rank size
///
/// \return     Original node or a node representing a reshape of the original.
///
ov::Output<ov::Node> reshape_channel_shaped_node_to_nchw(const ov::Output<ov::Node>& node,
                                                         const ov::Output<ov::Node>& expected_rank);

}  // namespace  reshape
}  // namespace onnx
}  // namespace frontend

namespace op {
namespace util {
/// \brief      Change shape of a value
///
/// \param[in]  value  The value to be reshaped.
/// \param[in]  shape  The new shape.
///
/// \return     Reshape:v1 op.
std::shared_ptr<Node> reshape(const Output<Node>& value, const Shape& shape);

/// \brief Permute axes according to specified axes_order parameter.
///
/// \param      The vlaue whose axes we want to permute.
/// \param      axes_order The permutation of axes.
///
/// \return     Transpose:v1 op.
std::shared_ptr<Node> reorder_axes(const Output<Node>& value, std::vector<size_t> axes_order = {});

/// \brief      Return transposed value (with axes in reversed order).
///
/// \param      Value to transpose.
///
/// \return     Transpose:v1 op.
std::shared_ptr<Node> transpose(const Output<Node>& value);

/// \brief       Flatten a value into a 2D matrix, with a static dividing axis.
///
/// \param       The tensor to be flattened.
/// \param       The axis dividing shape.
///
/// \return      The new value will be a 2D matrix representing the flattened input
/// node.
std::shared_ptr<Node> flatten(const Output<Node>& value, int axis);
}  // namespace util
}  // namespace op
}  // namespace ov
