// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <sstream>
#include <unordered_set>

#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "utils.hpp"

namespace ov::op::v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseFillEmptyRows* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);
    const auto& values_shape = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           values_shape.rank().compatible(1),
                           "The values input must be a 1D tensor.",
                           values_shape);

    const auto& dense_shape = input_shapes[1];
    const bool is_dense_shape_rank_dynamic = dense_shape.rank().is_dynamic();
    const bool is_dense_shape_valid =
        is_dense_shape_rank_dynamic || (dense_shape.size() == 1 && dense_shape[0].compatible(2));
    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        is_dense_shape_valid,
        "The dense_shape input must be 1D and have exactly 2 elements, meaning only 2D sparse tensors are supported.");

    const auto& indices_shape = input_shapes[2];
    const bool is_indices_shape_valid =
        indices_shape.rank().is_dynamic() || (indices_shape.size() == 2 && indices_shape[1].compatible(2) &&
                                              (is_dense_shape_rank_dynamic || values_shape.rank().is_dynamic() ||
                                               indices_shape[0].compatible(values_shape[0])));
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           is_indices_shape_valid,
                           "The indices input must be a 2D tensor with the first dimension matching the size of values "
                           "and the second dimension having 2 elements.",
                           indices_shape);

    const auto& default_value_shape = input_shapes[3];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           default_value_shape.rank().compatible(0),
                           "The default_value input must be a scalar.",
                           default_value_shape);

    auto output_shapes = std::vector<TRShape>(3);
    auto& output_indices_shape = output_shapes[0];
    auto& output_values_shape = output_shapes[1];
    auto& empty_row_indicator_shape = output_shapes[2];
    output_indices_shape.resize(2);
    output_values_shape.resize(1);
    empty_row_indicator_shape.resize(1);
    output_indices_shape[1] = 2;  // Only 2D cases are supported

    if (auto dense_shape_value = get_input_const_data_as_shape<TRShape>(op, 1, tensor_accessor);
        dense_shape_value && (*dense_shape_value).is_static()) {
        const auto& number_of_rows = (*dense_shape_value)[0].get_length();
        empty_row_indicator_shape[0] = number_of_rows;
        if (auto indices_value = get_input_const_data_as<TRShape, int64_t>(op, 2, tensor_accessor)) {
            auto is_valid_index = [](int64_t index, int64_t max_value) -> bool {
                return index >= 0 && index < max_value;
            };
            auto create_index_error_message =
                [](const std::string& dim_name, int64_t index, int64_t max_value) -> std::string {
                std::stringstream ss;
                ss << "Sparse tensor index out of bounds: " << dim_name << " " << index
                   << " is outside the valid range [0, " << (max_value - 1) << "]";
                return ss.str();
            };

            // Rows can be referenced multiple times in sparse representation
            std::unordered_set<int64_t> existing_rows;
            const auto& indices_data = *indices_value;
            const auto& number_of_cols = (*dense_shape_value)[1].get_length();
            for (size_t i = 0, i_next = 1; i_next < indices_data.size(); i += 2, i_next += 2) {
                auto row = indices_data[i];
                NODE_SHAPE_INFER_CHECK(op,
                                       input_shapes,
                                       is_valid_index(row, number_of_rows),
                                       create_index_error_message("row", row, number_of_rows));

                auto col = indices_data[i_next];
                NODE_SHAPE_INFER_CHECK(op,
                                       input_shapes,
                                       is_valid_index(col, number_of_cols),
                                       create_index_error_message("column", col, number_of_cols));

                existing_rows.insert(row);
            }

            using TDim = typename TRShape::value_type;
            TDim empty_rows_count = number_of_rows - existing_rows.size();
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
}  // namespace ov::op::v16
