// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/bucketize.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v3 {

template <class T>
void shape_infer(const Bucketize* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2) && output_shapes.size() == 1);

    const auto& data_shape = input_shapes[0];
    const auto& buckets_shape = input_shapes[1];

    NODE_VALIDATION_CHECK(op,
                          buckets_shape.rank().compatible(1),
                          "Buckets input must be a 1D tensor. Got: ",
                          buckets_shape);
    output_shapes[0] = data_shape;
}
}  // namespace v3
}  // namespace op
}  // namespace ov
