// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/selective_ssm.hpp"
#include "utils.hpp"

namespace ov::op::internal {

template <typename TDim>
bool merge_selective_ssm_dim(TDim& destination, bool& initialized, const TDim& source) {
    if (!initialized) {
        destination = source;
        initialized = true;
        return true;
    }
    return TDim::merge(destination, destination, source);
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const SelectiveSSM* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);

    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[0].rank().compatible(1));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[1].rank().compatible(3));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[2].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[3].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[4].rank().compatible(4));
    NODE_SHAPE_INFER_CHECK(op, input_shapes, input_shapes[5].rank().compatible(4));

    const auto& A_shape = input_shapes[0];
    const auto& dt_shape = input_shapes[1];
    const auto& B_shape = input_shapes[2];
    const auto& x_shape = input_shapes[3];
    const auto& C_shape = input_shapes[4];
    const auto& state_shape = input_shapes[5];
    using DimType = typename T::value_type;

    DimType batch_dim{};
    DimType sequence_dim{};
    DimType heads_dim{};
    DimType head_dim{};
    DimType groups_dim{};
    DimType state_size_dim{};

    bool batch_initialized = false;
    bool batch_ok = true;
    if (x_shape.rank().is_static())
        batch_ok &= merge_selective_ssm_dim(batch_dim, batch_initialized, x_shape[0]);
    if (dt_shape.rank().is_static())
        batch_ok &= merge_selective_ssm_dim(batch_dim, batch_initialized, dt_shape[0]);
    if (B_shape.rank().is_static())
        batch_ok &= merge_selective_ssm_dim(batch_dim, batch_initialized, B_shape[0]);
    if (C_shape.rank().is_static())
        batch_ok &= merge_selective_ssm_dim(batch_dim, batch_initialized, C_shape[0]);
    if (state_shape.rank().is_static())
        batch_ok &= merge_selective_ssm_dim(batch_dim, batch_initialized, state_shape[0]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           batch_ok,
                           "The batch dimension of `dt`, `B`, `x`, `C` and `recurrent_state` should be the same.");

    bool sequence_initialized = false;
    bool sequence_ok = true;
    if (x_shape.rank().is_static())
        sequence_ok &= merge_selective_ssm_dim(sequence_dim, sequence_initialized, x_shape[1]);
    if (dt_shape.rank().is_static())
        sequence_ok &= merge_selective_ssm_dim(sequence_dim, sequence_initialized, dt_shape[1]);
    if (B_shape.rank().is_static())
        sequence_ok &= merge_selective_ssm_dim(sequence_dim, sequence_initialized, B_shape[1]);
    if (C_shape.rank().is_static())
        sequence_ok &= merge_selective_ssm_dim(sequence_dim, sequence_initialized, C_shape[1]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           sequence_ok,
                           "The sequence length of `dt`, `B`, `x` and `C` should be the same.");

    bool heads_initialized = false;
    bool heads_ok = true;
    if (x_shape.rank().is_static())
        heads_ok &= merge_selective_ssm_dim(heads_dim, heads_initialized, x_shape[2]);
    if (A_shape.rank().is_static())
        heads_ok &= merge_selective_ssm_dim(heads_dim, heads_initialized, A_shape[0]);
    if (dt_shape.rank().is_static())
        heads_ok &= merge_selective_ssm_dim(heads_dim, heads_initialized, dt_shape[2]);
    if (state_shape.rank().is_static())
        heads_ok &= merge_selective_ssm_dim(heads_dim, heads_initialized, state_shape[1]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           heads_ok,
                           "The number of heads of `A`, `dt`, `x` and `recurrent_state` should be the same.");

    bool head_dim_initialized = false;
    bool head_dim_ok = true;
    if (x_shape.rank().is_static())
        head_dim_ok &= merge_selective_ssm_dim(head_dim, head_dim_initialized, x_shape[3]);
    if (state_shape.rank().is_static())
        head_dim_ok &= merge_selective_ssm_dim(head_dim, head_dim_initialized, state_shape[2]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           head_dim_ok,
                           "The head dimension of `x` and `recurrent_state` should be the same.");

    bool groups_initialized = false;
    bool groups_ok = true;
    if (B_shape.rank().is_static())
        groups_ok &= merge_selective_ssm_dim(groups_dim, groups_initialized, B_shape[2]);
    if (C_shape.rank().is_static())
        groups_ok &= merge_selective_ssm_dim(groups_dim, groups_initialized, C_shape[2]);
    NODE_SHAPE_INFER_CHECK(op, input_shapes, groups_ok, "The number of groups of `B` and `C` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           groups_dim.is_dynamic() || groups_dim.get_length() > 0,
                           "The number of groups must be greater than zero.");

    bool state_size_initialized = false;
    bool state_size_ok = true;
    if (state_shape.rank().is_static())
        state_size_ok &= merge_selective_ssm_dim(state_size_dim, state_size_initialized, state_shape[3]);
    if (B_shape.rank().is_static())
        state_size_ok &= merge_selective_ssm_dim(state_size_dim, state_size_initialized, B_shape[3]);
    if (C_shape.rank().is_static())
        state_size_ok &= merge_selective_ssm_dim(state_size_dim, state_size_initialized, C_shape[3]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_size_ok,
                           "The state size of `B`, `C` and `recurrent_state` should be the same.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_size_dim.is_dynamic() || state_size_dim.get_length() > 0,
                           "The state size must be greater than zero.");

    if (heads_dim.is_static() && groups_dim.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               groups_dim.get_length() != 0 && heads_dim.get_length() % groups_dim.get_length() == 0,
                               "The number of heads should be divisible by the number of groups.");
    }

    auto output_shapes = std::vector<TRShape>{x_shape, state_shape};
    if (x_shape.rank().is_static()) {
        output_shapes[0] = TRShape{batch_dim, sequence_dim, heads_dim, head_dim};
    }
    if (state_shape.rank().is_static()) {
        output_shapes[1] = TRShape{batch_dim, heads_dim, head_dim, state_size_dim};
    }
    return output_shapes;
}

}  // namespace ov::op::internal
