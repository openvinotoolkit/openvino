// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <rnn_cell_shape_inference.hpp>
#include "shape_infer_utils.hpp"

namespace ov {
namespace op {
namespace ShapeInferLSTM {
template <class OpsType, class ShapeType>
void lstm_shape_infer(const OpsType* op,
                      const std::vector<ShapeType>& input_shapes,
                      std::vector<ShapeType>& output_shapes,
                      std::size_t gates_count) {
    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;
    // If rank is dynamic, then output_shape is undefined
    for (const auto& input : input_shapes) {
        if (input.rank().is_dynamic()) {
            ShapeInfer::default_work(output_shapes[0]);
            ShapeInfer::default_work(output_shapes[1]);
            return;
        }
    }

    const auto& x_pshape = input_shapes[0];
    const auto& ht_pshape = input_shapes[1];
    const auto& ct_pshape = input_shapes[2];
    const auto& w_pshape = input_shapes[3];
    const auto& r_pshape = input_shapes[4];
    const auto& b_pshape = input_shapes[5];

    // Prepare OutShape
    auto& hidden_shape = output_shapes[0];
    auto& cell_shape = output_shapes[1];
    hidden_shape.resize(2);
    cell_shape.resize(2);

    // Check rnn common input
    util::validate_input_rank(dynamic_cast<const util::RNNCellBase*>(op),
                              std::vector<ShapeType>{x_pshape, ht_pshape, w_pshape, r_pshape, b_pshape});

    // Check cell
    NODE_VALIDATION_CHECK(op,
                          (ct_pshape.rank().is_static()),
                          "LSTMCell input tensor initial_cell_state shall have static rank.");
    NODE_VALIDATION_CHECK(op,
                          (ct_pshape.rank().get_length() == 2),
                          "LSTMCell input tensor initial_cell_state shall have dimension 2D.");
    // Check peepholes
    if (input_shapes.size() == 7) {
        const auto& p_pshape = input_shapes[6];
        NODE_VALIDATION_CHECK(op, (p_pshape.rank().is_static()), "LSTMCell input tensor P shall have static rank.");
        NODE_VALIDATION_CHECK(op,
                              (p_pshape.rank().get_length() == 1),
                              "LSTMCell input tensor P shall have dimension 1D.");
    }

    auto merged_batch_size = ht_pshape[0];
    auto merged_hidden_size = ht_pshape[1];

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    NODE_VALIDATION_CHECK(op,
                          DimType::merge(merged_batch_size, merged_batch_size, ht_pshape[0]) &&
                              DimType::merge(merged_batch_size, merged_batch_size, ct_pshape[0]) &&
                              DimType::merge(merged_batch_size, merged_batch_size, x_pshape[0]),
                          "Parameter batch_size not matched for X, initial_hidden_state or initial_cell_state "
                          "inputs.");

    // Merge hidden_size dimension across all inputs to evaluate output[1] dimension
    NODE_VALIDATION_CHECK(op,
                          DimType::merge(merged_hidden_size, merged_hidden_size, ht_pshape[1]) &&
                              DimType::merge(merged_hidden_size, merged_hidden_size, ct_pshape[1]) &&
                              DimType::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
                          "Parameter hidden_size not matched for R, initial_hidden_state and initial_cell_state "
                          "inputs.");

    // Validate hidden_size value for W, R and P inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  w_pshape[0].compatible(merged_hidden_size * gates_count),
                                  "Parameter hidden_size mistmatched in W input. Current value is: ",
                                  w_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * gates_count,
                                  ".");
        }

        if (r_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  r_pshape[0].compatible(merged_hidden_size * gates_count),
                                  "Parameter hidden_size mistmatched in R input. Current value is: ",
                                  r_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * gates_count,
                                  ".");
        }

        if (b_pshape[0].is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  b_pshape[0].compatible(merged_hidden_size * gates_count),
                                  "Parameter hidden_size mistmatched in B input. Current value is: ",
                                  b_pshape[0].get_length(),
                                  ", expected: ",
                                  merged_hidden_size.get_length() * gates_count,
                                  ".");
        }
    }

    hidden_shape[0] = merged_batch_size;
    hidden_shape[1] = merged_hidden_size;
    cell_shape[0] = merged_batch_size;
    cell_shape[1] = merged_hidden_size;
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
    if (p_pshape[0].is_static() && output_shapes[0][0].is_static()) {
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
