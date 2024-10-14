// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/search_sorted.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class TShape, class TRShape = result_shape_t<TShape>, class TDim = typename TShape::value_type>
std::vector<TRShape> shape_infer(const SearchSorted* op, const std::vector<TShape>& input_shapes) {
    const auto& sorted_shape = input_shapes[0];
    const auto& values_shape = input_shapes[1];
    const auto is_sorted_rank_static = sorted_shape.rank().is_static();
    const auto is_values_rank_static = values_shape.rank().is_static();
    TRShape output_shape;

    output_shape = is_values_rank_static ? values_shape : sorted_shape;
    if (is_sorted_rank_static) {
        const auto sorted_in_rank = sorted_shape.size();
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               sorted_in_rank > 0,
                               "The sorted sequence input cannot be a scalar.");
        if (sorted_in_rank == 1) {
            output_shape = values_shape;
            return {std::move(output_shape)};
        }
        else if (!is_values_rank_static) {
            output_shape[sorted_in_rank - 1] = Dimension::dynamic();
        } else {
            const auto values_in_rank = values_shape.size();
            NODE_SHAPE_INFER_CHECK(op,
                                input_shapes,
                                sorted_in_rank == values_in_rank,
                                "The inputs' ranks have to be compatible. If values input is a scalar the sorted sequence must be a 1D tensor.");
            for (size_t i = 0; i < sorted_in_rank - 1; ++i) {
                NODE_SHAPE_INFER_CHECK(op,
                                    input_shapes,
                                    TDim::merge(output_shape[i], sorted_shape[i], values_shape[i]),
                                    "All dimensions but the last one have to be compatible.");
            }
        }
    }
    return {std::move(output_shape)};
    }
}  // namespace v15
}  // namespace op
}  // namespace ov
