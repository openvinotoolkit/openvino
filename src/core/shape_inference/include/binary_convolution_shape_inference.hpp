// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/binary_convolution.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace convolution {
namespace validate {
/**
 * @brief Specific check of data shape for binary convolution data shape must be rank 4.
 *
 * The shape_infer is same as for Convolution operator except this check. @see convolution_shape_inference.hpp
 */
template <class TShape>
void data_shape(const v1::BinaryConvolution* op, const TShape& data_shape) {
    NODE_VALIDATION_CHECK(op, data_shape.rank().compatible(4), "Expected 4D for the input. Got: ", data_shape);
}
}  // namespace validate
}  // namespace convolution
}  // namespace op
}  // namespace ov
