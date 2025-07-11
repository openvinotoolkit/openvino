// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/sparse_fill_empty_rows_unpacked_string.hpp"
#include "utils.hpp"

namespace ov::op::v16 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SparseFillEmptyRowsUnpackedString* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 6);
    const auto& begins_shape = input_shapes[0];
    const auto& ends_shape = input_shapes[1];
    const auto& symbols_shape = input_shapes[2];
    const auto& indices_shape = input_shapes[3];
    const auto& dense_shape_shape = input_shapes[4];
    const auto& default_value_shape = input_shapes[5];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           begins_shape.compatible(ends_shape),
                           "The begins and ends inputs must have identical shapes.",
                           begins_shape,
                           ends_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           begins_shape.rank().compatible(1),
                           "The begins input must be a 1D tensor.",
                           begins_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           ends_shape.rank().compatible(1),
                           "The ends input must be a 1D tensor.",
                           ends_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           symbols_shape.rank().compatible(1),
                           "The symbols input must be a 1D tensor.",
                           symbols_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           indices_shape.rank().compatible(2),
                           "The indices input must be a 2D tensor.",
                           indices_shape);
    if (indices_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               indices_shape.rank().is_static() && indices_shape[1].compatible(2),
                               "The indices input must have second dimension equal to 2.",
                               indices_shape);
    }
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           dense_shape_shape.rank().compatible(1),
                           "The dense_shape input must be a 1D tensor.",
                           dense_shape_shape);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           default_value_shape.rank().compatible(1),
                           "The default_value input must be a 1D tensor.",
                           default_value_shape);
    if (begins_shape.rank().is_static() && indices_shape.rank().is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               begins_shape[0].compatible(indices_shape[0]),
                               "The begins and indices inputs must have the same shape.",
                               begins_shape,
                               indices_shape);
    }

    auto output_shapes = std::vector<TRShape>(5);
    auto& output_begins_shape = output_shapes[0];
    auto& output_ends_shape = output_shapes[1];
    auto& output_indices_shape = output_shapes[2];
    auto& output_symbols_shape = output_shapes[3];
    auto& empty_row_indicator_shape = output_shapes[4];

    output_begins_shape = begins_shape;
    output_ends_shape = ends_shape;
    output_indices_shape = indices_shape;
    output_symbols_shape = symbols_shape;
    empty_row_indicator_shape.resize(1);

    const auto& dense_shape_value = get_input_const_data_as<TRShape, int64_t>(op, 4, tensor_accessor);
    if (dense_shape_value) {
        const auto& dense_shape_data = *dense_shape_value;
        const size_t num_rows = dense_shape_data[0];
        empty_row_indicator_shape[0] = num_rows;

        const auto& indices_value = get_input_const_data_as<TRShape, int64_t>(op, 3, tensor_accessor);
        if (indices_value && symbols_shape.rank().is_static() && default_value_shape.rank().is_static() &&
            symbols_shape[0].is_static() && default_value_shape[0].is_static()) {
            const auto& indices_data = *indices_value;
            std::unordered_set<int64_t> filled_rows;
            for (size_t i = 0; i < indices_data.size() && i / 2 < indices_data.size() / 2; i += 2) {
                filled_rows.insert(indices_data[i]);
            }

            // Check if there's at least one empty row
            if (num_rows != filled_rows.size()) {
                const auto new_value_count = begins_shape[0].get_length() + num_rows - filled_rows.size();
                output_begins_shape[0] = new_value_count;
                output_ends_shape[0] = new_value_count;
                output_indices_shape[0] = new_value_count;
                output_symbols_shape[0] = symbols_shape[0].get_length() + default_value_shape[0].get_length();
            }
        }
    }
    return output_shapes;
}
}  // namespace ov::op::v16
