// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/search_sorted.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const SearchSorted* op, const std::vector<TShape>& input_shapes) {
    const auto& sorted_shape = input_shapes[0];
    const auto& values_shape = input_shapes[1];
    const auto is_sorted_rank_static = sorted_shape.rank().is_static();
    const auto is_values_rank_static = values_shape.rank().is_static();

    if (!is_sorted_rank_static || sorted_shape.size() == 1) {
        // If the sorted sequence is 1D, then any shape of the values input is allowed.
        // The shape of the output is the same as the shape of the values.
        return {values_shape};
    }

    const auto sorted_in_rank = sorted_shape.size();
    NODE_SHAPE_INFER_CHECK(op, input_shapes, sorted_in_rank > 0, "The sorted sequence input cannot be a scalar.");

    TRShape output_shape;
    if (!is_values_rank_static) {
        output_shape = sorted_shape;
        output_shape[sorted_in_rank - 1] = Dimension::dynamic();
    } else {
        output_shape = values_shape;
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            sorted_in_rank == values_shape.size(),
            "If the shape of sorted sequence is not 1D, the ranks of the inputs have to be compatible.");
        using TDim = typename TShape::value_type;
        for (size_t i = 0; i < sorted_in_rank - 1; ++i) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(output_shape[i], values_shape[i], sorted_shape[i]),
                                   "All dimensions but the last one have to be compatible.");
        }
    }

    return {std::move(output_shape)};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
