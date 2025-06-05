// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_set>

#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"
#include "utils.hpp"

namespace ov::op::v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseFillEmptyRowsUnpackedString* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);
    const auto& begins_shape = input_shapes[0];
    const auto& ends_shape = input_shapes[1];
    const auto& symbols_shape = input_shapes[2];
    const auto& default_value_shape = input_shapes[3];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           begins_shape.compatible(ends_shape),
                           "The begins and ends inputs must have identical shapes.",
                           begins_shape,
                           ends_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           symbols_shape.rank().compatible(1),
                           "The symbols input must be a 1D tensor.",
                           symbols_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           default_value_shape.rank().compatible(0),
                           "The default_value input must be a 1D tensor of u8 values.",
                           default_value_shape);

    auto output_shapes = std::vector<TRShape>(4);
    auto& output_begins_shape = output_shapes[0];
    auto& output_ends_shape = output_shapes[1];
    auto& output_symbols_shape = output_shapes[2];
    auto& empty_row_indicator_shape = output_shapes[3];
    
    output_begins_shape = begins_shape;
    output_ends_shape = ends_shape;
    output_symbols_shape = symbols_shape;
    empty_row_indicator_shape.resize(1);

    const auto& number_of_rows = begins_shape[0].get_length();
    if (begins_shape.rank().is_static() && begins_shape[0].is_static()) {
        empty_row_indicator_shape[0] = number_of_rows;
    }

    const auto& begins_value = get_input_const_data_as<TRShape, int64_t>(op, 0, tensor_accessor);
    const auto& ends_value = get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor);
    const auto& default_value_data = get_input_const_data_as<TRShape, uint8_t>(op, 3, tensor_accessor);

    if (begins_value && ends_value && default_value_data) {
        bool has_empty_rows = false;
        const auto& begins_data = *begins_value;
        const auto& ends_data = *ends_value;
        const auto& cols_per_row = begins_shape[1].get_length();

        for (int64_t row = 0; row < number_of_rows && !has_empty_rows; row++) {
            bool row_has_non_empty_string = false;
            for (int64_t col = 0; col < cols_per_row; col++) {
                const int64_t idx = row * cols_per_row + col;
                if (static_cast<size_t>(idx) < begins_data.size() && begins_data[idx] < ends_data[idx]) {
                    row_has_non_empty_string = true;
                    break;
                }
            }
            if (!row_has_non_empty_string) {
                has_empty_rows = true;
                break; // Stop checking as soon as we find one empty row
            }
        }

        if (has_empty_rows && symbols_shape[0].is_static()) {
            output_symbols_shape[0] = symbols_shape[0].get_length() + (*default_value_data).size();
        }
    }
    
    return output_shapes;
}
}  // namespace ov::op::v16
