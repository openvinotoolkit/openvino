// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "utils.hpp"

namespace ov::op::internal {
template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6, "Incorrect number of input for GatedDeltaNet");

    // batch, seq_len, head_num, head_size
    const auto& query_ps = input_shapes[0];
    const auto& key_ps = input_shapes[1];
    const auto& value_ps = input_shapes[2];
    const auto& state_ps = input_shapes[3];
    const auto& gate_ps = input_shapes[4];
    const auto& beta_ps = input_shapes[5];

    const auto q_head_num = query_ps[2];
    const auto k_head_num = key_ps[2];
    const auto v_head_num = value_ps[2];

    const auto k_head_size = key_ps[3];
    const auto q_head_size = query_ps[3];
    const auto v_head_size = value_ps[3];

    NODE_VALIDATION_CHECK(op,
                          q_head_num.compatible(k_head_num) && q_head_num.compatible(v_head_num),
                          "The number of heads in query key and value should be the same, but got ",
                          q_head_num,
                          " and ",
                          k_head_num,
                          ".");

    NODE_VALIDATION_CHECK(op,
                          k_head_size.compatible(q_head_size),
                          "The head size in key and query should be the same, but got ",
                          k_head_size,
                          " and ",
                          q_head_size,
                          ".");

    const auto gate_head_num = gate_ps[2];
    const auto beta_head_num = beta_ps[2];

    NODE_VALIDATION_CHECK(op,
                          gate_head_num.compatible(beta_head_num) && gate_head_num.compatible(q_head_num),
                          "The number of heads in gate, beta, and query should be the same, but got ",
                          gate_head_num,
                          " and ",
                          beta_head_num,
                          ".");

    // [batch, v_head_nums, k_head_size, v_head_size]
    const auto state_head_num = state_ps[1];
    const auto state_hidden_size_0 = state_ps[2];
    const auto state_hidden_size_1 = state_ps[3];
    NODE_VALIDATION_CHECK(op,
                          state_head_num.compatible(v_head_num),
                          "The number of heads in recurrent_state and value should be the same, but got ",
                          state_head_num,
                          " and ",
                          v_head_num,
                          ".");
    NODE_VALIDATION_CHECK(op,
                          state_hidden_size_0.compatible(k_head_size),
                          "The dim at shape[-2] of recurrent_state and head size of key should be the same, but got ",
                          state_hidden_size_0,
                          " and ",
                          k_head_size,
                          ".");
    NODE_VALIDATION_CHECK(op,
                          state_hidden_size_1.compatible(v_head_size),
                          "The dim at shape[-1] of recurrent_state and head size of value should be the same, but got ",
                          state_hidden_size_1,
                          " and ",
                          v_head_size,
                          ".");
    // output has the same shape and type as input value, output state has the same shape and type as input
    // recurrent_state
    auto output_shapes = std::vector<TRShape>{value_ps, state_ps};
    return output_shapes;
}
}  // namespace ov::op::internal
