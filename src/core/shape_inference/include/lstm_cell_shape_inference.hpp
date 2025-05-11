// Copyright (C) 2018-2025 Intel Corporation
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
namespace lstm_cell {
constexpr size_t gates_count = 4;
constexpr size_t num_state_nodes = 2;
constexpr size_t peepholes_count = 3;
}  // namespace lstm_cell

template <class T>
std::vector<result_shape_t<T>> shape_infer(const LSTMCell* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 7);

    auto output_shapes =
        rnn::cell_base_shape_infer(op, input_shapes, lstm_cell::gates_count, lstm_cell::num_state_nodes);
    const auto& hidden_size = output_shapes[0][1];
    if (hidden_size.is_dynamic()) {  // set hidden_size based on attribute
        output_shapes[0][1] = op->get_hidden_size();
        output_shapes[1][1] = op->get_hidden_size();
    }
    const auto& p_pshape = input_shapes[6];
    if (p_pshape[0].is_static() && hidden_size.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              p_pshape[0].compatible(hidden_size * 3),
                              "Parameter hidden_size mismatched in P input. Current value is: ",
                              p_pshape[0].get_length(),
                              ", expected: ",
                              hidden_size.get_length() * 3,
                              ".");
    }
    return output_shapes;
}
}  // namespace v0

namespace v4 {
namespace lstm_cell {
constexpr size_t gates_count = 4;
}

template <class TShape>
std::vector<result_shape_t<TShape>> shape_infer(const LSTMCell* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);
    constexpr auto num_state_nodes = 2;
    auto output_shapes = rnn::cell_base_shape_infer(op, input_shapes, lstm_cell::gates_count, num_state_nodes);
    if (output_shapes[0][1].is_dynamic()) {  // set hidden_size based on attribute
        output_shapes[0][1] = op->get_hidden_size();
        output_shapes[1][1] = op->get_hidden_size();
    }
    return output_shapes;
}
}  // namespace v4
}  // namespace op
}  // namespace ov
