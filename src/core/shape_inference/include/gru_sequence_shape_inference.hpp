// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/gru_sequence.hpp"
#include "rnn_base_shape_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v5 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ov::op::v5::GRUSequence* op, const std::vector<TShape>& input_shapes) {
    constexpr size_t expected_in_shapes_count = 6;
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == expected_in_shapes_count,
                          "Incorrect number of input shapes has been provided. Expected: ",
                          expected_in_shapes_count,
                          ", got: ",
                          input_shapes.size(),
                          ".");

    constexpr auto num_gates = 3;
    constexpr auto num_state_nodes = 1;
    return rnn::seq_base_shape_infer(op,
                                     input_shapes,
                                     num_gates,
                                     num_state_nodes,
                                     op->get_direction(),
                                     op->get_linear_before_reset());
}
}  // namespace v5
}  // namespace op
}  // namespace ov
