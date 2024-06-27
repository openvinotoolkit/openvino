// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/util/rnn_cell_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace rnn {
template <class TShape>
void validate_inputs_rank(const op::util::RNNCellBase* op,
                          const std::vector<TShape>& input_shapes,
                          const std::vector<Rank>& expected_ranks) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= expected_ranks.size(), "Can't validate inputs rank.");
    for (size_t i = 0; i < expected_ranks.size(); ++i) {
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

// Output shape layout:
// output_shapes[0...num_state_nodes]: [batch_size, hidden_size] // Rank always 2
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> cell_base_shape_infer(const op::util::RNNCellBase* op,
                                           const std::vector<TShape>& input_shapes,
                                           size_t num_gates,
                                           size_t num_state_nodes,
                                           bool linear_before_reset = false) {
    const auto num_inputs = 4 + num_state_nodes;
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= num_inputs, "Incorrect number of shapes has been provided.");

    std::vector<Rank> expected_in_ranks;
    expected_in_ranks.reserve(num_inputs);
    expected_in_ranks.insert(expected_in_ranks.end(), 1 + num_state_nodes, Rank(2));
    expected_in_ranks.insert(expected_in_ranks.end(), {2, 2, 1});

    rnn::validate_inputs_rank(op, input_shapes, expected_in_ranks);

    const auto& x_pshape = input_shapes[0];                    // [batch_size, input_size]
    const auto& ht_pshape = input_shapes[1];                   // [batch_size, hidden_size]
    const auto& w_pshape = input_shapes[1 + num_state_nodes];  // [3 * hidden_size, input_size]
    const auto& r_pshape = input_shapes[2 + num_state_nodes];  // [3 * hidden_size, hidden_size]
    const auto& b_pshape =
        input_shapes[3 + num_state_nodes];  // if linear_before_reset [4 * hidden_size], otherwise [3 * hidden_size]

    using DimType = typename TShape::value_type;

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    auto output_shapes = std::vector<TRShape>(1);
    output_shapes[0].push_back(x_pshape.rank().is_static() ? x_pshape[0] : DimType());
    output_shapes[0].push_back(ht_pshape.rank().is_static() ? ht_pshape[1] : DimType());
    auto& merged_batch_size = output_shapes[0][0];
    auto& merged_hidden_size = output_shapes[0][1];
    for (size_t i = 1; i <= num_state_nodes; ++i) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_batch_size,
                                             merged_batch_size,
                                             input_shapes[i].rank().is_static() ? input_shapes[i][0] : DimType()),
                              "Dimension `batch_size` is not matched between inputs.");
    }

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    // `hidden_size` attribute is not used for backward compatibility
    for (size_t i = 2; i <= num_state_nodes; ++i) {
        if (input_shapes[i].rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(merged_hidden_size, merged_hidden_size, input_shapes[i][1]),
                                  "Dimension `hidden_size` is not matched between inputs.");
        }
    }

    if (r_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_hidden_size, merged_hidden_size, r_pshape[1]),
                              "Dimension `hidden_size` is not matched between inputs.");
    }

    // Validate dimensions related to hidden_size for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  w_pshape[0].compatible(merged_hidden_size * num_gates),
                                  "First dimension of W input shape is required to be compatible with ",
                                  merged_hidden_size * num_gates,
                                  ". Got shape: ",
                                  w_pshape[0],
                                  ".");
        }

        if (r_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  r_pshape[0].compatible(merged_hidden_size * num_gates),
                                  "Fisrt dimension of R input shape is required to be compatible with ",
                                  merged_hidden_size * num_gates,
                                  ". Got shape: ",
                                  r_pshape[0],
                                  ".");
        }

        if (b_pshape.rank().is_static()) {
            auto bias_dim_multiplier = linear_before_reset ? (num_gates + 1) : num_gates;
            NODE_VALIDATION_CHECK(op,
                                  b_pshape[0].compatible(merged_hidden_size * bias_dim_multiplier),
                                  "First dimension of B input shape is required to be compatible with ",
                                  merged_hidden_size * bias_dim_multiplier,
                                  ". Got shape: ",
                                  b_pshape[0],
                                  ".");
        }
    } else {
        const size_t w_idx = 1 + num_state_nodes;
        for (size_t i = w_idx; i < w_idx + 2; ++i) {
            if (input_shapes[i].rank().is_static() && input_shapes[i][0].is_static()) {
                NODE_VALIDATION_CHECK(
                    op,
                    DimType::merge(merged_hidden_size, merged_hidden_size, input_shapes[i][0] / num_gates),
                    "Dimension `hidden_size` is not matched between inputs.");
            }
        }
    }

    output_shapes.resize(num_state_nodes, output_shapes[0]);
    return output_shapes;
}

// Output shapes layout:
// output_shapes[0]: [batch_size, num_directions, seq_length, hidden_size] // Rank always 4
// output_shapes[1... num_state_nodes]: [batch_size, num_directions, hidden_size] // Rank always 3
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> seq_base_shape_infer(const op::util::RNNCellBase* op,
                                          const std::vector<TShape>& input_shapes,
                                          size_t num_gates,
                                          size_t num_state_nodes,
                                          op::RecurrentSequenceDirection direction,
                                          bool linear_before_reset = false) {
    const auto num_inputs = 5 + num_state_nodes;
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= num_inputs, "Incorrect number of shapes has been provided.");

    std::vector<Rank> expected_in_ranks;
    expected_in_ranks.reserve(num_inputs);
    expected_in_ranks.insert(expected_in_ranks.end(), 1 + num_state_nodes, Rank(3));
    expected_in_ranks.insert(expected_in_ranks.end(), {1, 3, 3, 2});

    rnn::validate_inputs_rank(op, input_shapes, expected_in_ranks);

    const auto& x_pshape = input_shapes[0];
    const auto& ht_pshape = input_shapes[1];

    const auto& w_pshape = input_shapes[2 + num_state_nodes];
    const auto& r_pshape = input_shapes[3 + num_state_nodes];
    const auto& b_pshape = input_shapes[4 + num_state_nodes];

    using DimType = typename TShape::value_type;
    // Y output
    auto output_shapes = std::vector<TRShape>{{x_pshape.rank().is_static() ? x_pshape[0] : DimType(),
                                               DimType(),
                                               x_pshape.rank().is_static() ? x_pshape[1] : DimType(),
                                               ht_pshape.rank().is_static() ? ht_pshape[2] : DimType()}};

    // Merge batch_size dimension across all inputs to evaluate output[0] dimension
    auto& merged_batch_size = output_shapes[0][0];
    for (size_t i = 1; i <= 1 + num_state_nodes; ++i) {
        if (input_shapes[i].rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(merged_batch_size, merged_batch_size, input_shapes[i][0]),
                                  "Dimension `batch_size` is not matched between inputs.");
        }
    }

    // Merge hidden_size dimension across all inputs to evaluate output dimension
    // `hidden_size` attribute is not used for backward compatibility
    auto& merged_hidden_size = output_shapes[0][3];
    for (size_t i = 2; i <= num_state_nodes; ++i) {
        if (input_shapes[i].rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  DimType::merge(merged_hidden_size, merged_hidden_size, input_shapes[i][2]),
                                  "Dimension `hidden_size` is not matched between inputs.");
        }
    }

    if (r_pshape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              DimType::merge(merged_hidden_size, merged_hidden_size, r_pshape[2]),
                              "Dimension `hidden_size` is not matched between inputs.");
    }

    // Validate num_directions dimension across all inputs
    auto& merged_num_directions = output_shapes[0][1];
    const auto m_direction = direction;
    if (m_direction == op::RecurrentSequenceDirection::FORWARD ||
        m_direction == op::RecurrentSequenceDirection::REVERSE) {
        merged_num_directions = 1;
    } else if (m_direction == op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        merged_num_directions = 2;
    } else {
        NODE_VALIDATION_CHECK(op, false, "Attribute direction must be FORWARD or REVERSE or BIDIRECTIONAL.");
    }

    bool is_num_dir_valid = true;
    for (size_t i = 1; i <= num_state_nodes; ++i) {
        is_num_dir_valid &= DimType::merge(merged_num_directions,
                                           merged_num_directions,
                                           input_shapes[i].rank().is_static() ? input_shapes[i][1] : DimType());
    }

    for (size_t i = 2 + num_state_nodes; i < num_inputs; ++i) {
        is_num_dir_valid &= DimType::merge(merged_num_directions,
                                           merged_num_directions,
                                           input_shapes[i].rank().is_static() ? input_shapes[i][0] : DimType());
    }

    NODE_VALIDATION_CHECK(op,
                          is_num_dir_valid,
                          "Dimension `num_directions` doesn't match to other inputs or `direction` attribute.");

    // Validate dimensions related to hidden_size for W, R, B inputs
    if (merged_hidden_size.is_static()) {
        if (w_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  w_pshape[1].compatible(merged_hidden_size * num_gates),
                                  "Second dimension of W input shape is required to be compatible with ",
                                  merged_hidden_size * num_gates,
                                  ". Got shape: ",
                                  w_pshape[1],
                                  ".");
        }

        if (r_pshape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  r_pshape[1].compatible(merged_hidden_size * num_gates),
                                  "Second dimension of R input shape is required to be compatible with ",
                                  merged_hidden_size * num_gates,
                                  ". Got shape: ",
                                  r_pshape[1],
                                  ".");
        }

        if (b_pshape.rank().is_static()) {
            const auto bias_dim_multiplier = linear_before_reset ? (num_gates + 1) : num_gates;
            NODE_VALIDATION_CHECK(op,
                                  b_pshape[1].compatible(merged_hidden_size * bias_dim_multiplier),
                                  "Second dimension of B input shape is required to be compatible with ",
                                  merged_hidden_size * bias_dim_multiplier,
                                  ". Got shape: ",
                                  b_pshape[1],
                                  ".");
        }
    } else {
        const size_t w_idx = 2 + num_state_nodes;
        for (size_t i = w_idx; i < w_idx + 2; ++i) {
            if (input_shapes[i].rank().is_static() && input_shapes[i][0].is_static()) {
                NODE_VALIDATION_CHECK(
                    op,
                    DimType::merge(merged_hidden_size, merged_hidden_size, input_shapes[i][1] / num_gates),
                    "Dimension `hidden_size` is not matched between inputs.");
            }
        }
    }

    // Ho, Co outputs
    output_shapes.insert(output_shapes.end(),
                         num_state_nodes,
                         TRShape{merged_batch_size, merged_num_directions, merged_hidden_size});
    return output_shapes;
}
}  // namespace rnn
}  // namespace op
}  // namespace ov
