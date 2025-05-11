// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "openvino/frontend/visibility.hpp"
#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace frontend {

/// \brief Creates a random normal tensor with the given shape and type.
/// \details Uses Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param sizes Shape of the output tensor
/// \param target_type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
FRONTEND_API OutputVector make_random_normal(pass::NodeRegistry& registry,
                                       const Output<Node>& sizes,
                                       element::Type target_type,
                                       const Output<Node>& mean,
                                       const Output<Node>& scale,
                                       float seed);

/// \brief Creates a random normal tensor with the given shape and type.
/// \details Uses Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param sizes Shape of the output tensor
/// \param target_type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
FRONTEND_API std::pair<OutputVector, pass::NodeRegistry> make_random_normal(const Output<Node>& sizes,
                                                                      element::Type target_type,
                                                                      const Output<Node>& mean,
                                                                      const Output<Node>& scale,
                                                                      float seed);

}  // namespace frontend
}  // namespace ov
