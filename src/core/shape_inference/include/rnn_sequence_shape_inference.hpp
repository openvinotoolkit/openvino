// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/rnn_sequence.hpp"
#include "rnn_base_shape_inference.hpp"

namespace ov {
namespace op {
namespace v5 {
template <class TShape>
std::vector<TShape> shape_infer(const RNNSequence* op, const std::vector<TShape>& input_shapes) {
    constexpr auto num_gates = 1;
    constexpr auto num_state_nodes = 1;
    return rnn::rnn_seq_base_shape_infer(op, input_shapes, num_gates, num_state_nodes);
}
}  // namespace v5
}  // namespace op
}  // namespace ov
