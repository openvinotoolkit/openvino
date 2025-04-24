// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "openvino/op/matmul.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const MatMul* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& arg0_shape = input_shapes[0];
    const auto& arg1_shape = input_shapes[1];
    if (arg0_shape.rank().is_dynamic() || arg1_shape.rank().is_dynamic()) {
        return {ov::PartialShape::dynamic()};
    }

    auto output_shapes = std::vector<TRShape>();
    // ranks are known
    const bool transpose_a = op->get_transpose_a();
    const bool transpose_b = op->get_transpose_b();

    size_t arg0_rank = arg0_shape.size(), arg1_rank = arg1_shape.size();
    NODE_VALIDATION_CHECK(op, (arg0_rank != 0 && arg1_rank != 0), "Scalars are not supported as MatMul inputs.");

    // Temporary Dimension vectors to calculate output shape
    TRShape arg0_shape_tmp(arg0_shape), arg1_shape_tmp(arg1_shape);

    // 1. Applying transpositions specified by optional `transpose_a` and `transpose_b`
    // Only two right-most dimensions are swapped, other dimensions remain the same.
    // Transpose attributes are ignored for 1D tensors.
    if (transpose_a && arg0_rank > 1) {
        swap(arg0_shape_tmp[arg0_rank - 2], arg0_shape_tmp[arg0_rank - 1]);
    }
    if (transpose_b && arg1_rank > 1) {
        swap(arg1_shape_tmp[arg1_rank - 2], arg1_shape_tmp[arg1_rank - 1]);
    }

    // 2. One-dimensional tensors unsqueezing is applied to each input independently.
    if (arg0_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), 1);
        arg0_rank = arg0_shape_tmp.size();
    }
    if (arg1_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg1_shape_tmp.insert(arg1_shape_tmp.end(), 1);
        arg1_rank = arg1_shape_tmp.size();
    }
    // Check matrices dimensions compatibility,
    // COL_INDEX_DIM of the first matrix has to match ROW_INDEX_DIM of the second matrix.
    // Error is not thrown for dynamic dimensions bounds without intersection
    // to ensure MatMul backward compatibility.
    using DimType = typename T::value_type;
    auto merged_dimension = DimType();
    auto arg0_col_dim = arg0_shape_tmp[arg0_rank - 1];
    auto arg1_row_dim = arg1_shape_tmp[arg1_rank - 2];
    NODE_VALIDATION_CHECK(op,
                          DimType::merge(merged_dimension, arg0_col_dim, arg1_row_dim) || arg0_col_dim.is_dynamic() ||
                              arg1_row_dim.is_dynamic(),
                          "Incompatible MatMul matrix dimension. ",
                          "First input dimension=",
                          arg0_col_dim,
                          " at COL_INDEX_DIM=",
                          (arg0_rank - 1),
                          " doesn't match the second input dimension=",
                          arg1_row_dim,
                          " at ROW_INDEX_DIM=",
                          (arg1_rank - 2));
    // 3. If ranks of input arguments are different after steps 1 and 2,
    // the smaller tensor is unsqueezed from the left side of the shape
    // by necessary number of axes to make both shapes of the same rank.
    if (arg0_rank < arg1_rank)
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    else if (arg0_rank > arg1_rank)
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);
    // Both arg0_shape_tmp and arg1_shape_tmp have identical size now
    size_t max_rank = arg0_shape_tmp.size();
    std::vector<DimType> output_shape(max_rank);

    // 4. Usual rules of the broadcasting are applied for batch dimensions.
    // Broadcast all batches (last two dimensions represent matrix),
    // expand dim with value 1 to bigger dim if dimensions are not equal.
    for (size_t i = 0; i < max_rank - 2; ++i) {
        NODE_VALIDATION_CHECK(op,
                              DimType::broadcast_merge(output_shape[i], arg0_shape_tmp[i], arg1_shape_tmp[i]) ||
                                  arg0_shape_tmp[i].is_dynamic() || arg1_shape_tmp[i].is_dynamic(),
                              "Incompatible MatMul batch dimension. ",
                              "Can't merge first input dimension=",
                              arg0_shape_tmp[i],
                              " with second input dimension=",
                              arg1_shape_tmp[i],
                              " at index=",
                              i);
    }

    // In output_shape replace 2 last axes with ROW_INDEX_DIM from arg0 matrix
    // and COL_INDEX_DIM from arg1 matrix.
    output_shape[output_shape.size() - 2] = arg0_shape_tmp[arg0_shape_tmp.size() - 2];
    output_shape[output_shape.size() - 1] = arg1_shape_tmp[arg1_shape_tmp.size() - 1];

    // 5. Removing the temporary axes from originally 1D tensors.
    // Output shape of two 1D tensors multiplication will be a 0D tensor (scalar).
    if (arg0_shape.rank().get_length() == 1) {
        // arg0 input temporary axis inserted at ROW_INDEX_DIM is removed
        output_shape.erase(output_shape.begin() + (output_shape.size() - 2));
    }
    if (arg1_shape.rank().get_length() == 1) {
        // arg1 input temporary axis inserted at COL_INDEX_DIM is removed
        output_shape.erase(std::prev(output_shape.end()));
    }
    output_shapes.emplace_back(std::move(output_shape));
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
