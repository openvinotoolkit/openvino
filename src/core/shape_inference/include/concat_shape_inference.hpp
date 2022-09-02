// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/op/concat.hpp>

#include "utils.hpp"
namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const Concat* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() > 0 && output_shapes.size() == 1);
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    auto& output_pshape = output_shapes[0];
    DimType concatenation_axis_output_dim{0};
    output_pshape = input_shapes[0];
    const auto& output_rank = output_pshape.rank();
    int64_t axis = op->get_axis();
    if (output_rank.is_static()) {
        axis = axis < 0 ? axis + output_rank.get_length() : axis;
    } else {
        output_pshape = ov::PartialShape::dynamic();
        return;
    }

    for (uint64_t i = 0; i < input_shapes.size(); i++) {
        const auto& this_input_shape = input_shapes[i];
        const auto& this_input_rank = this_input_shape.rank();
        if (this_input_rank.is_static()) {

            auto concat_axis = axis;
            NODE_VALIDATION_CHECK(op,
                                  concat_axis < this_input_rank.get_length() && concat_axis >= 0
                                  && output_rank.get_length() == this_input_rank.get_length(),
                                  "Concatenation axis (",
                                  concat_axis,
                                  ") is out of bounds [",
                                  -this_input_rank.get_length(),
                                  ", ",
                                  this_input_rank.get_length() - 1,
                                  "] for ",
                                  "argument ",
                                  i,
                                  ", which has shape ",
                                  this_input_shape,
                                  ".");

            concatenation_axis_output_dim += this_input_shape[concat_axis];

            for (size_t i = 0; i < output_rank.get_length(); i++) {
                if (i == concat_axis)
                    continue;
                NODE_VALIDATION_CHECK(op,
                        output_pshape[i].compatible(this_input_shape[i]),
                        "Argument shapes are inconsistent; they must have the same rank, and must "
                        "have ",
                        "equal dimension everywhere except on the concatenation axis (axis ",
                        concat_axis,
                        ").");
            }
        } else {
            concatenation_axis_output_dim += Dimension::dynamic();
        }
    }

    if (output_pshape.rank().is_static()) {
        output_pshape[axis] = concatenation_axis_output_dim;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov