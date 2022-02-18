// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/reshape.hpp"
#include "ngraph/output_vector.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {

/// \brief Creates a random normal tensor with the given shape and type.
/// \details Uses Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param shape Shape of the output tensor
/// \param type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
OutputVector make_random_normal(const Output<ngraph::Node>& shape,
                                element::Type type,
                                float mean,
                                float scale,
                                float seed);

}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
