// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "ov_ops/augru_sequence.hpp"
#include "rnn_base_shape_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace internal {
template <class ShapeType, class TRShape = result_shape_t<ShapeType>>
std::vector<TRShape> shape_infer(const ov::op::internal::AUGRUSequence* op,
                                 const std::vector<ShapeType>& input_shapes) {
    constexpr size_t expected_in_shapes_count = 7;
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == expected_in_shapes_count,
                          "Incorrect number of input shapes has been provided. Expected: ",
                          expected_in_shapes_count,
                          ", got: ",
                          input_shapes.size(),
                          ".");

    constexpr auto num_gates = 3;
    constexpr auto num_state_nodes = 1;
    auto output_shapes = rnn::seq_base_shape_infer(op,
                                                   input_shapes,
                                                   num_gates,
                                                   num_state_nodes,
                                                   op->get_direction(),
                                                   op->get_linear_before_reset());

    // A input shape validation // [batch_size, seq_length, 1]
    const auto& a_shape = input_shapes.back();
    const auto& x_shape = input_shapes[0];
    NODE_VALIDATION_CHECK(op, a_shape.rank().compatible(3), "'A' input must be a 3D tensor.");
    if (a_shape.rank().is_static()) {
        if (x_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  x_shape.rank().get_length() > 1 && a_shape[0].compatible(x_shape[0]),
                                  "Dimension `batch_size` must be the same for `X` and `A` inputs.");
            NODE_VALIDATION_CHECK(op,
                                  x_shape.rank().get_length() > 2 && a_shape[1].compatible(x_shape[1]),
                                  "Dimension `seq_length` must be the same for `X` and `A` inputs.");
        }
        NODE_VALIDATION_CHECK(op, a_shape[2].compatible(1), "The last dimension of `A` shape must be equal to `1`.");
    }
    return output_shapes;
}
}  // namespace internal
}  // namespace op
}  // namespace ov
