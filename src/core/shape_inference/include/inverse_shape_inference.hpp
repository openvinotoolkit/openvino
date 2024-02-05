// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/inverse.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v14 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Inverse* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto input_rank = input_shape.rank();
    if (input_rank.is_static()) {
        const auto& input_shape = input_shapes[0];

        const auto size = input_shape.size();
        NODE_VALIDATION_CHECK(op, size >= 2, "Input must be at least a 2D matrix.");

        const auto& const_dims = get_input_const_data_as_shape<TRShape>(op, 0, ta);
        if (const_dims) {
            NODE_VALIDATION_CHECK(op,
                                  (*const_dims)[size - 2].get_min_length() == (*const_dims)[size - 1].get_min_length(),
                                  "Input must contain square matrices of the same shape.");
        }

        return {input_shape};
    } else {
        return {ov::PartialShape::dynamic()};
    }
}
}  // namespace v14
}  // namespace op
}  // namespace ov
