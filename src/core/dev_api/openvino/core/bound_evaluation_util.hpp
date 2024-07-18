// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {

/// \brief Checks if bounds can be propagated on Node output.
/// \param output  Node output to test for bound propagation.
/// \param order   return vector of nodes for output which can be processed for bound evaluation.
/// \return True if bounds can be propagated for output and order vector has valid data, otherwise false.
OPENVINO_API bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

namespace util {

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
}  // namespace util
}  // namespace ov
