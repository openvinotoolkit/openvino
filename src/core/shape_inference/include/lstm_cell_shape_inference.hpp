// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/lstm_cell.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace ShapeInferLSTM {
template <class OpsType, class ShapeType>
void lstm_shape_infer(const OpsType* op,
                      const std::vector<ShapeType>& input_shapes,
                      std::vector<ShapeType>& output_shapes,
                      std::size_t gates_count) {
    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;
    enum { X, initial_hidden_state, initial_cell_state, W, R, B };
    std::vector<bool> input_rank_static(6, false);
    bool all_rank_dynamic = true;
    bool all_rank_static = true;
    // Prepare OutShape
    auto& hidden_shape = output_shapes[0];
    auto& cell_shape = output_shapes[1];
    hidden_shape.resize(2);
    cell_shape.resize(2);

    // If rank is dynamic, then output_shape is undefined
    for (size_t i = 0; i < input_shapes.size() && i < 6; i++) {
        input_rank_static[i] = input_shapes[i].rank().is_static();
        all_rank_dynamic = all_rank_dynamic && !input_rank_static[i];
        all_rank_static = all_rank_static && input_rank_static[i];
    }

    if (all_rank_dynamic) {
        return;
    }
    const auto& x_pshape = input_shapes[0];
    const auto& w_pshape = input_shapes[3];

    DimType output_batch_size;
    DimType output_hidden_size;
    bool is_batch_init = false;
    bool is_hidden_init = false;

    // deduce batch/hidden_size
    for (size_t i = 0; i < input_shapes.size() && i < 6; i++) {
        const auto& input = input_shapes[i];
        if (input_rank_static[i]) {
            // batch could be deduced from x, cell_state or hidden_state
            if (i == X || i == initial_cell_state || i == initial_hidden_state) {
                NODE_VALIDATION_CHECK(op,
                                      (input.size() == 2),
                                      "LSTMCell input rank is not correct for ",
                                      i,
                                      " input parameter. Current rank: ",
                                      input.size(),
                                      ", expected: 2.");
                if (!is_batch_init) {
                    output_batch_size = input[0];
                    is_batch_init = true;
                } else {
                    NODE_VALIDATION_CHECK(
                        op,
                        DimType::merge(output_batch_size, output_batch_size, input[0]),
                        "Parameter batch_size not matched for X, initial_hidden_state or initial_cell_state "
                        "inputs.");
                }
                if (i == initial_cell_state || i == initial_hidden_state) {
                    if (!is_hidden_init) {
                        output_hidden_size = input[1];
                        is_hidden_init = true;
                    } else {
                        NODE_VALIDATION_CHECK(op,
                                              DimType::merge(output_hidden_size, output_hidden_size, input[1]),
                                              "Parameter hidden_size not matched for W, R, B, initial_hidden_state and "
                                              "initial_cell_state "
                                              "inputs.");
                    }
                }
            } else if (i == W || i == R || i == B) {
                // check input dimension
                if (i == B) {
                    NODE_VALIDATION_CHECK(op,
                                          (input.size() == 1),
                                          "LSTMCell input tensor dimension is not correct for ",
                                          i,
                                          " input parameter. Current input length: ",
                                          input.size(),
                                          ", expected: 1.");
                    if (input[0].is_static()) {
                        if (!is_hidden_init) {
                            output_hidden_size = input[0].get_length() / gates_count;
                            is_hidden_init = true;
                        } else {
                            NODE_VALIDATION_CHECK(
                                op,
                                DimType::merge(output_hidden_size,
                                               output_hidden_size,
                                               input[0].get_length() / gates_count),
                                "Parameter hidden_size not matched for W, R, B, initial_hidden_state and "
                                "initial_cell_state "
                                "inputs.");
                        }
                    }
                } else {
                    NODE_VALIDATION_CHECK(op,
                                          (input.size() == 2),
                                          "LSTMCell input rank is not correct for ",
                                          i,
                                          " input parameter. Current rank: ",
                                          input.size(),
                                          ", expected: 2.");
                    if (input[0].is_static()) {
                        if (!is_hidden_init) {
                            output_hidden_size = input[0].get_length() / gates_count;
                            is_hidden_init = true;
                        } else {
                            NODE_VALIDATION_CHECK(
                                op,
                                DimType::merge(output_hidden_size,
                                               output_hidden_size,
                                               input[0].get_length() / gates_count),
                                "Parameter hidden_size not matched for W, R, B, initial_hidden_state and "
                                "initial_cell_state "
                                "inputs.");
                        }
                    }
                    if (i == R) {
                        if (!is_hidden_init) {
                            output_hidden_size = input[1];
                            is_hidden_init = true;
                        } else {
                            NODE_VALIDATION_CHECK(op,
                                                  DimType::merge(output_hidden_size, output_hidden_size, input[1]),
                                                  "Parameter hidden_size not matched for W, R, B, initial_hidden_state "
                                                  "and initial_cell_state "
                                                  "inputs.");
                        }
                    }
                }
            }
        }
    }
    // Check peepholes
    if (input_shapes.size() == 7) {
        const auto& p_pshape = input_shapes[6];
        NODE_VALIDATION_CHECK(op, (p_pshape.rank().compatible(1)), "LSTMCell input tensor P shall have dimension 1D.");
    }

    // check input size
    if (input_rank_static[X] && input_rank_static[W]) {
        NODE_VALIDATION_CHECK(op, (x_pshape[1].compatible(w_pshape[1])), "LSTMCell mismatched input_size dimension.");
    }

    hidden_shape[0] = output_batch_size;
    hidden_shape[1] = output_hidden_size;
    cell_shape[0] = output_batch_size;
    cell_shape[1] = output_hidden_size;
}

}  // namespace ShapeInferLSTM

namespace v0 {
using ShapeInferLSTM::lstm_shape_infer;
template <class T>
void shape_infer(const LSTMCell* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 7 && output_shapes.size() == 2);
    const auto& p_pshape = input_shapes[6];

    lstm_shape_infer(op, input_shapes, output_shapes, op->s_gates_count);
    const auto& hidden_size = output_shapes[0][1];
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
using ShapeInferLSTM::lstm_shape_infer;
template <class T>
void shape_infer(const LSTMCell* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6 && output_shapes.size() == 2);
    lstm_shape_infer(op, input_shapes, output_shapes, op->s_gates_count);
}
}  // namespace v4
}  // namespace op
}  // namespace ov
