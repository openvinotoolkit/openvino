// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseFillEmptyRows* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    const auto& values_shape = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           values_shape.rank().compatible(1),
                           "The values input must be a 1D tensor. Got: ",
                           values_shape);

    const auto& dense_shape = input_shapes[1];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dense_shape.rank().compatible(1),
                           "The dense_shape input must be a 1D tensor. Got: ",
                           dense_shape);
    if (dense_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            dense_shape[0].compatible(2),
            "The dense_shape input must have exactly 2 elements. Only 2D sparse tensors are supported.");
    }

    const auto& indices_shape = input_shapes[2];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           indices_shape.rank().compatible(2),
                           "The indices input must be a 2D tensor. Got: ",
                           indices_shape);

    if (indices_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            indices_shape[1].compatible(2),
            "The indices_shape's second dimension must have 2 elements. Only 2D sparse tensors are supported.");
        if (values_shape.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   indices_shape[0].compatible(values_shape[0]),
                                   "The first dimension of indices must match the size of values.");
        }
    }

    const auto& default_value_shape = input_shapes[3];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           default_value_shape.rank().compatible(0),
                           "The default_value input must be a scalar. Got: ",
                           default_value_shape);

    auto output_shapes = std::vector<TRShape>(3);
    auto& output_indices_shape = output_shapes[0];
    auto& output_values_shape = output_shapes[1];
    auto& empty_row_indicator_shape = output_shapes[2];
    output_indices_shape.resize(2);
    output_values_shape.resize(1);
    empty_row_indicator_shape.resize(1);
    output_indices_shape[1] = 2;  // Only 2D cases are supported

    if (auto dense_shape_value = get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor)) {
        const auto& number_of_rows = (*dense_shape_value)[0].get_length();
        empty_row_indicator_shape[0] = number_of_rows;

        if (auto indices_value = get_input_const_data_as<TRShape, int64_t>(op, 2, tensor_accessor)) {
            // Rows can be referenced multiple times in sparse representation
            std::unordered_set<int64_t> existing_rows;
            const auto& indices_data = *indices_value;
            size_t indices_count = indices_data.size() / 2;
            const auto& number_of_cols = (*dense_shape_value)[1].get_length();
            for (size_t i = 0; i < indices_count; ++i) {
                int64_t row = indices_data[i * 2];
                NODE_SHAPE_INFER_CHECK(op,
                                       input_shapes,
                                       row >= 0 && row < static_cast<int64_t>(number_of_rows),
                                       "Sparse tensor index out of bounds: row ",
                                       row,
                                       " is outside the valid range [0, ",
                                       number_of_rows - 1,
                                       "]");
                int64_t col = indices_data[i * 2 + 1];
                NODE_SHAPE_INFER_CHECK(op,
                                       input_shapes,
                                       col >= 0 && col < static_cast<int64_t>(number_of_cols),
                                       "Sparse tensor index out of bounds: column ",
                                       col,
                                       " is outside the valid range [0, ",
                                       number_of_cols - 1,
                                       "]");
                existing_rows.insert(row);
            }
            int64_t empty_rows_count = number_of_rows - existing_rows.size();
            output_indices_shape[0] = indices_shape[0] + empty_rows_count;
            output_values_shape[0] = values_shape[0] + empty_rows_count;
        } else {
            output_indices_shape[0] = Dimension::dynamic();
            output_values_shape[0] = Dimension::dynamic();
        }
    } else {
        empty_row_indicator_shape[0] = Dimension::dynamic();
    }

    return output_shapes;
}
}  // namespace v16
}  // namespace op
}  // namespace ov
