// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

/// \brief Checks if all the elements of the bound Tensor are positive
bool tensor_is_positive(const Tensor& bound);

/// \brief Estimates upper bound for node output tensors using only upper bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of Tensors representing resulting upper value estimations
/// \return boolean status if value evaluation was successful.
bool default_upper_bound_evaluator(const Node* node, TensorVector& output_values);

/// \brief Estimates lower bound for node output tensors using only lower bounds of the nodes
/// inputs.
/// \param node Operation to be performed
/// \param output_values Vector of Tensors representing resulting lower value estimations
/// \return boolean status if value evaluation was successful.
bool default_lower_bound_evaluator(const Node* node, TensorVector& output_values);

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param Node output pointing to the tensor for estimation.
/// \return Tensor to estimated value.
Tensor evaluate_lower_bound(const Output<Node>& output);

/// \brief Evaluates lower value estimation of the output tensor. Traverses graph up to deduce
/// estimation through it.
/// \param output Tensor to be estimated.
/// \return Tensor to estimated value.
Tensor evaluate_upper_bound(const Output<Node>& output);

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
bool interval_bound_evaluator(const Node* node, TensorVector& lower_output_values, TensorVector& upper_output_values);
}  // namespace ov
