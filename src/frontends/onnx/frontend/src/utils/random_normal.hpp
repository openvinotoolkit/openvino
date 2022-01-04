// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace onnx_import {
namespace detail {

/// \brief Creates a random normal tensor with the given shape and type.
/// \details Uses Box-Mueller algorithm to generate random numbers from a Gauassian distribution
/// \param shape Shape of the output tensor
/// \param type Type of the output tensor
/// \param mean Mean of the distribution
/// \param scale Standard deviation of the distribution
/// \param seed Seed for the random number generator
OutputVector make_random_normal(const Output<ov::Node>& shape,
                                element::Type type,
                                float mean,
                                float scale,
                                float seed);

}  // namespace detail
}  // namespace onnx_import
}  // namespace ov
