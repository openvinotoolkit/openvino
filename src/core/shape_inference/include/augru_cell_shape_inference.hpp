// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "gru_cell_shape_inference.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace internal {
template <class ShapeType>
void shape_infer(const ov::op::internal::AUGRUCell* op,
                 const std::vector<ShapeType>& input_shapes,
                 std::vector<ShapeType>& output_shapes) {
    constexpr size_t expected_in_shapes_count = 6;
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == expected_in_shapes_count,
                          "Incorrect number of input shapes has been provided. Expected: ",
                          expected_in_shapes_count,
                          ", got: ",
                          input_shapes.size(),
                          ".");

    rnn::gru_cell_shape_infer(op, input_shapes, output_shapes);

    // `A` input shape validation // [batch_size, 1]
    const auto& a_shape = input_shapes.back();
    const auto& x_shape = input_shapes[0];
    NODE_VALIDATION_CHECK(op, a_shape.rank().compatible(2), "'A' input must be a 2D tensor.");
    if (a_shape.rank().is_static()) {
        if (x_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  x_shape.rank().get_length() > 1 && a_shape[0].compatible(x_shape[0]),
                                  "Dimension `batch_size` must be the same for `X` and `A` inputs.");
        }
        NODE_VALIDATION_CHECK(op, a_shape[1].compatible(1), "The last dimension of `A` shape must be equal to `1`.");
    }
}
}  // namespace internal
}  // namespace op
}  // namespace ov
