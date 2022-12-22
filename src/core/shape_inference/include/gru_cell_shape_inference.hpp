// Copyright (C) 2018-2022 Intel Corporation
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
namespace rnn {

// Output shape layout:
// output_shapes[0]: [batch_size, hidden_size] // Rank always 2
template <class OpType, class ShapeType>
void gru_cell_shape_infer(const OpType* op,
                          const std::vector<ShapeType>& input_shapes,
                          std::vector<ShapeType>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() >= 5 && output_shapes.size() == 1,
                          "Incorrect number of shapes has been provided.");

    auto& y_out_shape = output_shapes[0];
    y_out_shape.resize(2);  // Rank always 2

    rnn::validate_inputs_rank(op, input_shapes, {2, 2, 2, 2, 1});

    const auto& x_pshape = input_shapes[0];   // [batch_size, input_size]
    const auto& ht_pshape = input_shapes[1];  // [batch_size, hidden_size]
    const auto& w_pshape = input_shapes[2];   // [3 * hidden_size, input_size]
    const auto& r_pshape = input_shapes[3];   // [3 * hidden_size, hidden_size]
    const auto& b_pshape = input_shapes[4];   // if linear_before_reset [4 * hidden_size], otherwise [3 * hidden_size]

    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    DimType merged_batch_size = x_pshape.rank().is_static() ? x_pshape[0] : DimType();
    NODE_VALIDATION_CHECK(
        op,
        DimType::merge(merged_batch_size, merged_batch_size, ht_pshape.rank().is_static() ? ht_pshape[0] : DimType()),
        "Dimension `batch_size` is not matched between inputs.");

    // Set batch_size dimension
    y_out_shape[0] = merged_batch_size;

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    // `hidden_size` attribute is not used for backward compatibility
    DimType merged_hidden_size = ht_pshape.rank().is_static() ? ht_pshape[1] : DimType();
    NODE_VALIDATION_CHECK(
        op,
        DimType::merge(merged_hidden_size, merged_hidden_size, r_pshape.rank().is_static() ? r_pshape[1] : DimType()),
        "Dimension `hidden_size` is not matched between inputs.");

    // Validate dimensions related to hidden_size for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        constexpr auto gru_gates_count = 3;
        if (w_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  w_pshape[0].compatible(merged_hidden_size * gru_gates_count),
                                  "First dimension of W input shape is required to be compatible with ",
                                  merged_hidden_size * gru_gates_count,
                                  ". Got shape: ",
                                  w_pshape[0],
                                  ".");
        }

        if (r_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  r_pshape[0].compatible(merged_hidden_size * gru_gates_count),
                                  "Fisrt dimension of R input shape is required to be compatible with ",
                                  merged_hidden_size * gru_gates_count,
                                  ". Got shape: ",
                                  r_pshape[0],
                                  ".");
        }

        if (b_pshape.rank().is_static()) {
            auto bias_dim_multiplier = op->get_linear_before_reset() ? (gru_gates_count + 1) : gru_gates_count;
            NODE_VALIDATION_CHECK(op,
                                  b_pshape[0].compatible(merged_hidden_size * bias_dim_multiplier),
                                  "First dimension of B input shape is required to be compatible with ",
                                  merged_hidden_size * bias_dim_multiplier,
                                  ". Got shape: ",
                                  b_pshape[0],
                                  ".");
        }
    }

    // Set hidden_size dimension
    y_out_shape[1] = merged_hidden_size;
}
}  // namespace rnn
namespace v3 {
template <class ShapeType>
void shape_infer(const ov::op::v3::GRUCell* op,
                 const std::vector<ShapeType>& input_shapes,
                 std::vector<ShapeType>& output_shapes) {
    rnn::gru_cell_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v3
}  // namespace op
}  // namespace ov
