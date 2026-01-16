// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"

namespace ov {

/// \brief Checks if bounds can be propagated on Node output.
/// \param output  Node output to test for bound propagation.
/// \param order   return vector of nodes for output which can be processed for bound evaluation.
/// \return True if bounds can be propagated for output and order vector has valid data, otherwise false.
OPENVINO_API bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

namespace util {
/**
 * @ingroup ov_runtime_attr_api
 * @brief ForceInvalidation class represents runtime info attribute that forces bounds invalidation
 * even when SkipInvalidation is set. Used when input source changes and bounds need recalculation.
 * The attribute is automatically removed after invalidation occurs.
 */
class OPENVINO_API ForceInvalidation : public RuntimeAttribute {
public:
    OPENVINO_RTTI("ForceInvalidation", "0", RuntimeAttribute);
    ForceInvalidation() = default;
    ~ForceInvalidation();
    bool is_copyable() const override {
        return false;
    }
};

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

/// \brief Compares bounds (lower and upper values) of two tensors.
/// \param lhs First tensor to compare.
/// \param rhs Second tensor to compare.
/// \return True if both tensors have the same lower and upper bound values, false otherwise.
OPENVINO_API bool have_same_bounds(const descriptor::Tensor& lhs, const descriptor::Tensor& rhs);

/// \brief Sets ForceInvalidation attribute on tensor to force bounds invalidation on next invalidate_values() call.
/// \param tensor Tensor to set the ForceInvalidation attribute on.
OPENVINO_API void set_force_invalidation(ov::descriptor::Tensor& tensor);
}  // namespace util
}  // namespace ov
