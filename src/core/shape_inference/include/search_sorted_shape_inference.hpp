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
    const auto sorted_shape_rank = sorted_shape.rank();
    const auto values_shape_rank = values_shape.rank();

    if (sorted_shape_rank.is_static()) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               sorted_shape_rank.get_length() > 0,
                               "The sorted sequence input cannot be a scalar.");
    }

    TRShape output_shape;
    if (values_shape_rank.is_dynamic() && sorted_shape_rank.is_static()) {
        output_shape = sorted_shape;
        output_shape[sorted_shape.size() - 1] = Dimension::dynamic();
    } else {
        output_shape = values_shape;
    }

    if (values_shape_rank.is_static() && values_shape_rank == 0) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               sorted_shape_rank == 1,
                               "If values input is a scalar the sorted sequence must be a 1D tensor.");
    }

    if (sorted_shape_rank.is_static() && values_shape_rank.is_static() && sorted_shape_rank.get_length() > 1) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               sorted_shape.size() == values_shape.size(),
                               "The inputs' ranks have to be compatible.");
        for (size_t i = 0; i < sorted_shape.size() - 1; ++i) {
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   TDim::merge(output_shape[i], sorted_shape[i], values_shape[i]),
                                   "All dimensions but the last one have to be compatible.");
        }
    }

    return {std::move(output_shape)};
}
}  // namespace v15
}  // namespace op
}  // namespace ov
