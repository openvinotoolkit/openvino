// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/core/validation_util.hpp>
#include <openvino/op/gru_sequence.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace rnn {
template <class OpType, class ShapeType>
void validate_inputs_rank(const OpType* op,
                          const std::vector<ShapeType>& input_shapes,
                          const std::vector<Rank>& expected_ranks) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= expected_ranks.size(), "Can't validate inputs rank.");
    for (auto i = 0; i < expected_ranks.size(); ++i) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[i].rank().compatible(expected_ranks[i]),
                              "Shape rank of input at ",
                              i,
                              " is incompatible. Expected rank: ",
                              expected_ranks[i],
                              ", actual shape: ",
                              input_shapes[i],
                              ".");
    }
}

// Output shapes layout:
// output_shapes[0]: [batch_size, num_directions, seq_length, hidden_size] // Rank always 4
// output_shapes[1]: [batch_size, num_directions, hidden_size] // Rank always 3
template <class OpType, class ShapeType>
void gru_sequence_shape_infer(const OpType* op,
                              const std::vector<ShapeType>& input_shapes,
                              std::vector<ShapeType>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() >= 6 && output_shapes.size() == 2,
                          "Incorrect number of shapes has been provided.");

    auto& y_out_shape = output_shapes[0];
    auto& ho_out_shape = output_shapes[1];
    y_out_shape.resize(4);   // Rank always 4
    ho_out_shape.resize(3);  // Rank always 3

    rnn::validate_inputs_rank(op, input_shapes, {3, 3, 1, 3, 3, 2});

    const auto& x_pshape = input_shapes[0];
    const auto& ht_pshape = input_shapes[1];
    const auto& sl_pshape = input_shapes[2];
    const auto& w_pshape = input_shapes[3];
    const auto& r_pshape = input_shapes[4];
    const auto& b_pshape = input_shapes[5];

    using DimType = typename std::iterator_traits<typename ShapeType::iterator>::value_type;

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    DimType merged_batch_size = x_pshape.rank().is_static() ? x_pshape[0] : DimType();
    NODE_VALIDATION_CHECK(
        op,
        DimType::merge(merged_batch_size, merged_batch_size, ht_pshape.rank().is_static() ? ht_pshape[0] : DimType()) &&
            DimType::merge(merged_batch_size,
                           merged_batch_size,
                           sl_pshape.rank().is_static() ? sl_pshape[0] : DimType()),
        "Dimension `batch_size` is not matched between inputs.");

    // Set batch_size dimension
    y_out_shape[0] = merged_batch_size;
    ho_out_shape[0] = merged_batch_size;

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    // `hidden_size` attribute is not used for backward compatibility
    DimType merged_hidden_size = ht_pshape.rank().is_static() ? ht_pshape[2] : DimType();
    NODE_VALIDATION_CHECK(op,
                          DimType::merge(merged_hidden_size,
                                         merged_hidden_size,
                                         ht_pshape.rank().is_static() ? ht_pshape[2] : DimType()) &&
                              DimType::merge(merged_hidden_size,
                                             merged_hidden_size,
                                             r_pshape.rank().is_static() ? r_pshape[2] : DimType()),
                          "Dimension `hidden_size` is not matched between inputs.");

    // Validate num_directions dimension across all inputs
    size_t valid_num_directions;
    const auto m_direction = op->get_direction();
    if (m_direction == op::RecurrentSequenceDirection::FORWARD ||
        m_direction == op::RecurrentSequenceDirection::REVERSE) {
        valid_num_directions = 1;
    } else if (m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        valid_num_directions = 2;
    } else {
        NODE_VALIDATION_CHECK(op, false, "Attribute direction must be FORWARD or REVERSE or BIDIRECTIONAL.");
    }

    DimType merged_num_directions = DimType(valid_num_directions);
    NODE_VALIDATION_CHECK(op,
                          DimType::merge(merged_num_directions,
                                         merged_num_directions,
                                         ht_pshape.rank().is_static() ? ht_pshape[1] : DimType()) &&
                              DimType::merge(merged_num_directions,
                                             merged_num_directions,
                                             w_pshape.rank().is_static() ? w_pshape[0] : DimType()) &&
                              DimType::merge(merged_num_directions,
                                             merged_num_directions,
                                             r_pshape.rank().is_static() ? r_pshape[0] : DimType()) &&
                              DimType::merge(merged_num_directions,
                                             merged_num_directions,
                                             b_pshape.rank().is_static() ? b_pshape[0] : DimType()),
                          "Dimension `num_directions` doesn't match to other inputs or `direction` attribute.");

    // Set num_directions dimension
    y_out_shape[1] = merged_num_directions;
    ho_out_shape[1] = merged_num_directions;

    // Set seq_len dimension
    y_out_shape[2] = x_pshape.rank().is_static() ? x_pshape[1] : DimType();

    // Validate dimensions related to hidden_size for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        constexpr auto gru_seq_gates_count = 3;
        if (w_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  w_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                                  "Second dimension of W input shape is required to be compatible with ",
                                  merged_hidden_size * gru_seq_gates_count,
                                  ". Got shape: ",
                                  w_pshape[1],
                                  ".");
        }

        if (r_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  r_pshape[1].compatible(merged_hidden_size * gru_seq_gates_count),
                                  "Second dimension of R input shape is required to be compatible with ",
                                  merged_hidden_size * gru_seq_gates_count,
                                  ". Got shape: ",
                                  r_pshape[1],
                                  ".");
        }

        if (b_pshape.rank().is_static()) {
            auto bias_dim_multiplier = op->get_linear_before_reset() ? (gru_seq_gates_count + 1) : gru_seq_gates_count;
            NODE_VALIDATION_CHECK(op,
                                  b_pshape[1].compatible(merged_hidden_size * bias_dim_multiplier),
                                  "Second dimension of B input shape is required to be compatible with ",
                                  merged_hidden_size * bias_dim_multiplier,
                                  ". Got shape: ",
                                  b_pshape[1],
                                  ".");
        }
    }

    // Set hidden_size dimension
    y_out_shape[3] = merged_hidden_size;
    ho_out_shape[2] = merged_hidden_size;
}
}  // namespace rnn
namespace v5 {
template <class ShapeType>
void shape_infer(const ov::op::v5::GRUSequence* op,
                 const std::vector<ShapeType>& input_shapes,
                 std::vector<ShapeType>& output_shapes) {
    constexpr size_t expected_in_shapes_count = 6;
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == expected_in_shapes_count,
                          "Incorrect number of input shapes has been provided. Expected: ",
                          expected_in_shapes_count,
                          ", got: ",
                          input_shapes.size(),
                          ".");

    rnn::gru_sequence_shape_infer(op, input_shapes, output_shapes);
}
}  // namespace v5
}  // namespace op
}  // namespace ov
