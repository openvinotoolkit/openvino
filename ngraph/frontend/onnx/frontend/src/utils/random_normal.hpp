// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/reshape.hpp"
#include "ngraph/output_vector.hpp"

namespace ngraph {
namespace onnx_import {
namespace detail {

/// \brief Generates random normal tensor with the given shape and type.
OutputVector make_random_normal(const Output<ngraph::Node>& shape, element::Type type, float mean, float scale);

/// \brief Creates a random normal tensor with the given shape and type.
OutputVector make_random_normal(const Output<ngraph::Node>& shape,
                                element::Type type,
                                float mean,
                                float scale,
                                float seed);

/// \brief Use Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param shape Shape of the output tensor
/// \param type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
/// \param global_seed Global seed for the random number generator
OutputVector box_muller(const Output<ngraph::Node>& shape,
                        element::Type type,
                        float mean,
                        float scale,
                        uint64_t op_seed = 0,
                        uint64_t global_seed = 0);

}  // namespace detail
}  // namespace onnx_import
}  // namespace ngraph
