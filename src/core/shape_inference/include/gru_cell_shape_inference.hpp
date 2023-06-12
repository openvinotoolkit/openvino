// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/gru_cell.hpp>

#include "gru_cell_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
template <class TShape>
std::vector<TShape> shape_infer(const GRUCell* op, const std::vector<TShape>& input_shapes) {
    constexpr auto num_gates = 3;
    constexpr auto num_state_nodes = 1;
    return rnn::rnn_cell_base_shape_infer(op, input_shapes, num_gates, num_state_nodes, op->get_linear_before_reset());
}

template <class TShape>
void shape_infer(const GRUCell* op, const std::vector<TShape>& input_shapes, std::vector<TShape>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}
}  // namespace v3
}  // namespace op
}  // namespace ov
