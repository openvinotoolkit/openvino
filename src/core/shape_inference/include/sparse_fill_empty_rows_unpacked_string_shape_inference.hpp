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
                           default_value_shape.rank().compatible(1),
                           "The default_value input must be a 1D tensor.",
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

    if (begins_shape.rank().is_static()) {
        empty_row_indicator_shape[0] = begins_shape[0];
    }

    const auto& begins_value = get_input_const_data_as<TRShape, int64_t>(op, 0, tensor_accessor);
    const auto& ends_value = get_input_const_data_as<TRShape, int64_t>(op, 1, tensor_accessor);

    if (begins_value && ends_value && symbols_shape.rank().is_static() && symbols_shape[0].is_static()) {
        const auto& begins_data = *begins_value;
        const auto& ends_data = *ends_value;
        const auto cols_per_row = begins_shape[1].get_length();
        using TVal = typename TShape::value_type::value_type;

        for (TVal row = 0; row < begins_shape[0].get_length(); row++) {
            bool row_has_non_empty_string = false;
            const TVal row_start_idx = row * cols_per_row;

            for (TVal col = 0; col < cols_per_row; col++) {
                const TVal idx = row_start_idx + col;
                if (static_cast<size_t>(idx) < begins_data.size() && begins_data[idx] < ends_data[idx]) {
                    row_has_non_empty_string = true;
                    break;
                }
            }

            if (!row_has_non_empty_string) {
                output_symbols_shape[0] = symbols_shape[0].get_length() + default_value_shape[0].get_length();
                break;  // No reason to keep checking other rows, we store the default value only once
            }
        }
    }
    return output_shapes;
}
}  // namespace ov::op::v16
