// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "gru_cell_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "openvino/op/gru_cell.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GRUCell* op, const std::vector<TShape>& input_shapes) {
    constexpr auto num_gates = 3;
    constexpr auto num_state_nodes = 1;
    return rnn::cell_base_shape_infer(op, input_shapes, num_gates, num_state_nodes, op->get_linear_before_reset());
}
}  // namespace v3
}  // namespace op
}  // namespace ov
