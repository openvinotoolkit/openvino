// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace descriptor {
class Tensor;
}  // namespace descriptor

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

/// \brief Checks if two outputs have the same bounds (lower and upper values).
/// Returns true if both have no bounds, or if both have bounds with equal values.
/// \note Unlike internal are_equal(), returns true when BOTH outputs have no bounds
///       (semantically: "no bounds == no bounds").
/// \param lhs First output to compare bounds from.
/// \param rhs Second output to compare bounds to.
/// \return True if bounds are equivalent, false otherwise.
OPENVINO_API bool have_same_bounds(const Output<Node>& lhs, const Output<Node>& rhs);

/// \brief Sets force invalidation on tensor bounds, bypassing SkipInvalidation attribute.
/// Temporarily removes SkipInvalidation, calls invalidate_values(), then restores it.
/// \param tensor Tensor descriptor to invalidate bounds on.
OPENVINO_API void set_bounds_to_invalidate(descriptor::Tensor& tensor);

}  // namespace util
}  // namespace ov
