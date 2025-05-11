// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/rnn_cell.hpp"
#include "rnn_base_shape_inference.hpp"

namespace ov {
namespace op {
namespace v0 {
template <class TShape>
std::vector<result_shape_t<TShape>> shape_infer(const RNNCell* op, const std::vector<TShape>& input_shapes) {
    constexpr auto num_gates = 1;
    constexpr auto num_state_nodes = 1;
    return rnn::cell_base_shape_infer(op, input_shapes, num_gates, num_state_nodes);
}
}  // namespace v0
}  // namespace op
}  // namespace ov
