// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/bucketize.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Bucketize* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2));

    const auto& data_shape = input_shapes[0];
    const auto& buckets_shape = input_shapes[1];

    NODE_VALIDATION_CHECK(op,
                          buckets_shape.rank().compatible(1),
                          "Buckets input must be a 1D tensor. Got: ",
                          buckets_shape);
    return {data_shape};
}
}  // namespace v3
}  // namespace op
}  // namespace ov
