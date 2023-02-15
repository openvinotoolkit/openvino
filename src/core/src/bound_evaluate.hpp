// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "bound_evaluation_util.hpp"

namespace ov {
// bool could_propagate(const Output<Node>& output, std::vector<Node*>& order);

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

/// \brief Checks if lower and upper bounds of the corresponding tensor are set (not nullptr)
/// and pointers are the same. It doesn't check if lower and upper values are the same relying
/// only on pointers comparison.
bool has_and_set_equal_bounds(const Output<Node>& source);

/// \brief Checks if all node's inputs [first, last] have set bounds.
///
/// \param node
/// \param first_idx   Index of first node for check.
/// \param last_idx    Index of last node for check.
///
/// \return True If all inputs between [first, last] have bounds set, otherwise false. Return false if last input
/// greater than node's inputs count.
bool have_node_inputs_bounds_set(const ov::Node* const node, const size_t first_idx, const size_t last_idx);

}  // namespace ov
