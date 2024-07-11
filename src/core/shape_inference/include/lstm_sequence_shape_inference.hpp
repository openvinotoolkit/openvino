// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/lstm_sequence.hpp"
#include "rnn_base_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
OPENVINO_SUPPRESS_DEPRECATED_START
template <class TShape>
std::vector<result_shape_t<TShape>> shape_infer(const LSTMSequence* op, const std::vector<TShape>& input_shapes) {
    OPENVINO_SUPPRESS_DEPRECATED_END
    constexpr auto num_gates = 4;
    constexpr auto num_state_nodes = 2;
    const auto output_shapes =
        rnn::seq_base_shape_infer(op, input_shapes, num_gates, num_state_nodes, op->get_direction());
    // Validate rank and dimension for P input (the input doesn't exists in the next version of LSTM or other RNN based
    // ops) The checks are compatible with the original restrictions of the v0::LSTMSequence
    const auto& hidden_size = output_shapes[0][3];
    if (input_shapes.size() > 7 && input_shapes[7].is_static() && hidden_size.is_static()) {
        const auto& p_pshape = input_shapes[7];
        NODE_VALIDATION_CHECK(op, p_pshape.rank().compatible(2), "Input tensor P should have rank equal 2.");
        NODE_VALIDATION_CHECK(op,
                              p_pshape[1].compatible(hidden_size * (num_gates - 1)),
                              "Inorrect shape of P input. Second dimension is: ",
                              p_pshape[1],
                              ", expected: ",
                              hidden_size.get_length() * (num_gates - 1),
                              ".");
    }
    return output_shapes;
}
}  // namespace v0
namespace v5 {
template <class TShape>
std::vector<result_shape_t<TShape>> shape_infer(const LSTMSequence* op, const std::vector<TShape>& input_shapes) {
    constexpr auto num_gates = 4;
    constexpr auto num_state_nodes = 2;
    return rnn::seq_base_shape_infer(op, input_shapes, num_gates, num_state_nodes, op->get_direction());
}
}  // namespace v5
}  // namespace op
}  // namespace ov
