// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "rnn_base_shape_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace v0 {
template <class T>
void shape_infer(const LSTMCell* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 7 && output_shapes.size() == 2);
    constexpr auto num_state_nodes = 2;
    output_shapes = rnn::rnn_cell_base_shape_infer(op, input_shapes, op->s_gates_count, num_state_nodes);
    const auto& hidden_size = output_shapes[0][1];
    const auto& p_pshape = input_shapes[6];
    if (p_pshape[0].is_static() && hidden_size.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              p_pshape[0].compatible(hidden_size * op->s_peepholes_count),
                              "Parameter hidden_size mistmatched in P input. Current value is: ",
                              p_pshape[0].get_length(),
                              ", expected: ",
                              hidden_size.get_length() * op->s_peepholes_count,
                              ".");
    }
}
}  // namespace v0

namespace v4 {
template <class TShape>
std::vector<TShape> shape_infer(const LSTMCell* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);
    constexpr auto num_gates = 4;
    constexpr auto num_state_nodes = 2;
    auto output_shapes = rnn::rnn_cell_base_shape_infer(op, input_shapes, num_gates, num_state_nodes);
    if (output_shapes[0][1].is_dynamic()) {  // set hidden_size based on attribute
        output_shapes[0][1] = op->get_hidden_size();
        output_shapes[1][1] = op->get_hidden_size();
    }
    return output_shapes;
}

template <class T>
void shape_infer(const LSTMCell* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}
}  // namespace v4
}  // namespace op
}  // namespace ov
