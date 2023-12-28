// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
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
